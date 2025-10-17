# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from fsspec import AbstractFileSystem
from fsspec.asyn import AsyncFileSystem

logger = logging.getLogger(__name__)

_SAFETENSOR_DTYPE_MAP: Dict[str, np.dtype] = {
    "F16": np.dtype("float16"),
    "BF16": np.dtype(jnp.bfloat16),
    "F32": np.dtype("float32"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"),
    "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"),
    "BOOL": np.dtype("bool"),
}


ShardingFunction = Callable[[Tuple[int, ...]], Optional[jax.sharding.Sharding]]


@dataclass(frozen=True)
class TensorRecord:
    key: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    file_path: str
    byte_start: int
    byte_end: int
    fs: AsyncFileSystem

    async def get_slice(self, index) -> np.ndarray:
        """Fetch the requested slice, coalescing to a single contiguous range read."""

        if not isinstance(index, tuple):
            index = (index,)

        if Ellipsis in index:
            if index.count(Ellipsis) > 1:
                raise IndexError("only a single ellipsis is allowed in indexing")
            ellipsis_pos = index.index(Ellipsis)
            remaining = len(self.shape) - (len(index) - 1)
            index = index[:ellipsis_pos] + (slice(None),) * remaining + index[ellipsis_pos + 1 :]

        if len(index) < len(self.shape):
            index = index + (slice(None),) * (len(self.shape) - len(index))

        if len(index) != len(self.shape):
            raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")

        axis_entries: List[np.ndarray] = []
        axis_kinds: List[str] = []
        zero_extent = False

        for dim, idx in zip(self.shape, index):
            if isinstance(idx, slice):
                start, stop, step = idx.indices(dim)
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                if step < 0:
                    raise NotImplementedError("negative slice steps are not supported")

                length = 0 if stop <= start else (stop - start + (step - 1)) // step
                if length == 0:
                    zero_extent = True

                axis_entries.append(np.arange(start, start + length * step, step, dtype=np.int64))
                axis_kinds.append("slice")
            elif isinstance(idx, (int, np.integer)):
                pos = int(idx)
                if pos < 0:
                    pos += dim
                if pos < 0 or pos >= dim:
                    raise IndexError(f"index {idx} is out of bounds for axis with size {dim}")
                axis_entries.append(np.array([pos], dtype=np.int64))
                axis_kinds.append("index")
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        if zero_extent:
            output_shape = tuple(len(arr) for arr, kind in zip(axis_entries, axis_kinds) if kind == "slice")
            return np.empty(output_shape, dtype=self.dtype)

        if not axis_entries:
            raw = await self.fs._cat_file(self.file_path, start=self.byte_start, end=self.byte_end)
            return np.frombuffer(raw, dtype=self.dtype, count=1).reshape(())

        strides: List[int] = []
        stride = 1
        for dim in reversed(self.shape):
            strides.append(stride)
            stride *= dim
        strides.reverse()

        mesh = np.meshgrid(*axis_entries, indexing="ij", sparse=False)
        linear_indices = np.zeros(mesh[0].shape, dtype=np.int64)
        for coord, stride_val in zip(mesh, strides):
            linear_indices += coord * stride_val

        flat_indices = linear_indices.ravel()
        min_index = int(flat_indices.min())
        max_index = int(flat_indices.max())

        byte_start = self.byte_start + min_index * self.dtype.itemsize
        byte_end = self.byte_start + (max_index + 1) * self.dtype.itemsize

        raw = await self.fs._cat_file(self.file_path, start=byte_start, end=byte_end)
        buffer = np.frombuffer(raw, dtype=self.dtype)

        offsets = flat_indices - min_index
        gathered = buffer.take(offsets)
        result = gathered.reshape(linear_indices.shape)

        for axis, kind in reversed(list(enumerate(axis_kinds))):
            if kind == "index":
                result = np.take(result, indices=0, axis=axis)

        return np.asarray(result, dtype=self.dtype, copy=False)

    async def read(self) -> np.ndarray:
        """Fetch the full tensor."""

        raw = await self.fs._cat_file(self.file_path, start=self.byte_start, end=self.byte_end)
        array = np.frombuffer(raw, dtype=self.dtype)
        if self.shape:
            return array.reshape(self.shape)
        else:
            return array.reshape(())


class _AsyncifyingFileSystemWrapper(AsyncFileSystem):
    """Wrap a synchronous AbstractFileSystem to provide async methods using a thread pool."""

    def __init__(self, fs: AbstractFileSystem):
        super().__init__()
        self._fs = fs
        import concurrent.futures

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._fs.cat_file(path, start=start, end=end, **kwargs),
        )


# See https://huggingface.co/docs/safetensors/en/index#format for format spec
# It's pretty simple:
# # - 8 bytes: little-endian uint64 header length N
# # - N bytes: UTF-8 JSON header of shapes/dtypes/data offsets
# # - remaining bytes: raw tensor data blobs
async def _read_metadata_async(fs: AsyncFileSystem, path: str) -> Dict[str, TensorRecord]:
    header_len_bytes = await fs._cat_file(path, start=0, end=8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    metadata_bytes = await fs._cat_file(path, start=8, end=8 + header_len)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    tensors: Dict[str, TensorRecord] = {}
    data_offset_base = 8 + header_len

    for key, meta in metadata.items():
        if key == "__metadata__":
            continue
        dtype_name: str = meta["dtype"]
        dtype = _SAFETENSOR_DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")

        rel_start, rel_end = meta["data_offsets"]
        tensors[key] = TensorRecord(
            key=key,
            dtype=dtype,
            shape=tuple(meta["shape"]),
            file_path=path,
            byte_start=data_offset_base + rel_start,
            byte_end=data_offset_base + rel_end,
            fs=fs,
        )

    return tensors


def _apply_dtype_override(array: np.ndarray, dtype_override: Optional[jnp.dtype]) -> np.ndarray:
    if dtype_override is None:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.dtype(dtype_override), copy=False)
    return array


async def _materialize_unsharded_tensor(
    record: TensorRecord,
    dtype_override: Optional[jnp.dtype],
) -> jax.Array:
    array = await record.read()
    array = _apply_dtype_override(array, dtype_override)
    return jnp.asarray(array)


async def _materialize_sharded_tensor(
    record: TensorRecord,
    sharding: jax.sharding.Sharding,
    dtype_override: Optional[jnp.dtype],
) -> jax.Array:
    indices_map = sharding.devices_indices_map(record.shape)
    local_devices = sharding.addressable_devices

    async def _fetch(device):
        indices = tuple(indices_map[device])
        tensor_slice = await record.get_slice(indices)
        tensor_slice = _apply_dtype_override(tensor_slice, dtype_override)
        return jax.device_put(tensor_slice, device)

    per_device_arrays = await asyncio.gather(*(_fetch(device) for device in local_devices))
    return jax.make_array_from_single_device_arrays(record.shape, sharding, per_device_arrays)


async def read_safetensors_fsspec(
    path: str,
    *,
    dtype_override: Optional[jnp.dtype] = None,
    fs: Optional[AbstractFileSystem] = None,
    sharding_fn: Optional[ShardingFunction] = None,
) -> Dict[str, jax.Array]:
    """
    Stream tensors from a safetensors file using fsspec, optionally sharding the outputs.
    In the future we could make this lazy, but for now we eagerly materialize everything, which is the only way
    we use it currently.

    Args:
        path: The fsspec-compatible path to the safetensors file.
        dtype_override: If provided, floating-point tensors will be cast to this dtype.
        fs: An optional fsspec filesystem to use. If not provided, one will be created based on the path protocol.
        sharding_fn: An optional function that takes a tensor shape and returns a jax.sharding.Sharding
            object to use for that tensor, or None for no sharding.
    """

    protocol, fs_path = fsspec.core.split_protocol(path)
    if protocol is None:
        protocol = "file"
        fs_path = path

    if fs is None:
        fs = fsspec.filesystem(protocol, asynchronous=True, anon=False)

    if isinstance(fs, AsyncFileSystem):
        async_fs = fs
    else:
        async_fs = _AsyncifyingFileSystemWrapper(fs)

    target_path = fs_path if fs_path is not None else path
    async_tensors = await _read_metadata_async(async_fs, target_path)

    async def _materialize(key: str, record: TensorRecord):
        sharding = sharding_fn(record.shape) if sharding_fn is not None else None
        if sharding is not None:
            return key, await _materialize_sharded_tensor(record, sharding, dtype_override)
        array = await _materialize_unsharded_tensor(record, dtype_override)
        return key, array

    tasks = [_materialize(key, record) for key, record in async_tensors.items()]
    results = await asyncio.gather(*tasks)
    return {key: value for key, value in results}
