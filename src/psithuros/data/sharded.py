import itertools
from math import prod
from typing import TypeVar, Iterable, Sequence, Tuple, Iterator

import numpy as np
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import PartitionSpec

from psithuros.data.text import IndexedDataset, TokenizedDocumentCache
from psithuros.mesh import MeshInfo

In = TypeVar("In")
Ex = TypeVar("Ex")

# TODO: maybe generify this to work on more than just single sequence inputs
# TODO: write tests to verify this works when data spans multiple processes
# ExampleShape = Union[Tuple[int, ...], Sequence[Tuple[int, ...]]]

# This is hard for me to think about.

# We are using GlobalDeviceArray to coordinate data loading. GlobalDeviceArrays are constructed using
# from_batched_callback, which passes in a list of slices that correspond to the entries in the GDA. The GDA is as of
# size (num_micro_batches, microbatch_size, seq_len). The device mesh is (data, model), and we want to replicate data
# across the model axis. We partition the above array as (None, data, None), meaning that the minibatch axis is
# distributed across the data axis. Each process is responsible for loading data for its devices. GDA's
# from_batched_callback will tell us how much data to load and for which device, so we just have to load it.
# We do need to make sure that, in the event the data axis is larger than num_devices_per_process, each process that
# is part of the same position in the device mesh

class ShardedIndexedDataset(Iterable[GlobalDeviceArray]):
    def __init__(self,
                 doc_cache: TokenizedDocumentCache,
                 mesh_info: MeshInfo,
                 seq_len: int,
                 microbatched: bool = True
                 ):
        self.mesh_info = mesh_info
        self.microbatched = microbatched
        process_data_pos = self.mesh_info.process_mesh_position[0]
        num_data_process_groups = self.mesh_info.process_mesh_size[0]

        assert num_data_process_groups <= self.mesh_info.process_count

        self.indexed_dataset = IndexedDataset(doc_cache, seq_len, stride=None).shard(
            process_data_pos,
            num_data_process_groups,
        )

    def __iter__(self) -> Iterator[GlobalDeviceArray]:
        # TODO: support not infinite iterators
        def loop_gen():
            while True:
                for ex in self.indexed_dataset:
                    yield ex

        it = loop_gen()

        batch_shape = self.batch_shape()
        if self.microbatched:
            pspec = PartitionSpec(None, self.mesh_info.data_axis_name, None)
        else:
            pspec = PartitionSpec(self.mesh_info.data_axis_name, None)

        assert len(batch_shape) == len(pspec)

        def callback(indices: Sequence[Tuple[slice, ...]]):
            # TODO: it seems like we may want to just directly index into the tokenized dataset somehow. This seems a bit
            # more fragile
            # there is one entry in indices per device. They may be identical.
            # convert slices to tuples so we can use hashes
            out = []
            data_for_group = {}
            for index_group in indices:
                my_indices = tuple(s.indices(axis_size) for axis_size, s in zip(batch_shape, index_group))
                assert (s[2] == 1 for s in my_indices)
                slice_sizes = [s[1] - s[0] for s in my_indices]
                num_examples = prod(slice_sizes[0:-1])
                if my_indices not in data_for_group:
                    data_for_group[my_indices] = np.stack(list([ex['input_ids'] for ex in itertools.islice(it, num_examples)])).reshape(*slice_sizes)
                out.append(data_for_group[my_indices])

            return out

        while True:
            yield GlobalDeviceArray.from_batched_callback(
                batch_shape,
                self.mesh_info.mesh,
                pspec,
                callback,
            )

    def batch_shape(self):
        if self.microbatched:
            return (self.mesh_info.microbatches_per_step, self.mesh_info.microbatch_size, self.indexed_dataset.seq_len)
        else:
            return (self.mesh_info.batch_size, self.indexed_dataset.seq_len)

