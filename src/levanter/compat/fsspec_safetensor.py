# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import struct
from collections import OrderedDict

import fsspec
import numpy as np
import tensorstore as ts


class _AsyncLRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.locks = {}

    async def get_or_wait(self, key, fetch_fn):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self.cache:
                return self.cache[key]
            result = await fetch_fn()
            self.cache[key] = result
            self.cache.move_to_end(key)
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
            del self.locks[key]
            return result


class _AsyncFsspecReader:
    def __init__(self, gcs_path: str, cache_size=128):
        self.gcs_path = gcs_path
        protocol = fsspec.core.split_protocol(gcs_path)[0]
        if protocol is None:
            protocol = "file"
        self.fs = fsspec.filesystem(protocol, asynchronous=True, anon=False)
        self.cache = _AsyncLRUCache(cache_size)

    async def read_range(self, start: int, length: int) -> bytes:
        key = (start, length)

        async def fetch():
            end = start + length
            if hasattr(self.fs, "_cat_file"):
                return await self.fs._cat_file(self.gcs_path, start=start, end=end)
            # have to be slow
            # return await self.fs._cat_file(self.gcs_path, start=start, end=end)
            return self.fs.cat_file(self.gcs_path, start=start, end=end)

        return await self.cache.get_or_wait(key, fetch)


async def _load_safetensors_metadata(reader: _AsyncFsspecReader):
    """Parses metadata directly from a safetensors file using fsspec."""
    header_len_bytes = await reader.read_range(0, 8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)

    # Read the JSON metadata
    metadata_bytes = await reader.read_range(8, header_len)
    json_str = metadata_bytes.decode("utf-8")
    metadata = json.loads(json_str)
    data_offset_base = 8 + header_len

    tensors = {}
    for k, v in metadata.items():
        if k == "__metadata__":
            continue
        rel_start, rel_end = v["data_offsets"]
        tensors[k] = {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "offset": (data_offset_base + rel_start, data_offset_base + rel_end),
        }

    return tensors


def _coalesce_reads(offsets, itemsize, max_gap=128):
    """Group byte offsets into contiguous or nearly-contiguous ranges."""
    offsets = sorted(set(offsets))
    ranges = []
    current_start = offsets[0]
    current_end = current_start + itemsize

    for off in offsets[1:]:
        if off <= current_end + max_gap:
            current_end = max(current_end, off + itemsize)
        else:
            ranges.append((current_start, current_end))
            current_start = off
            current_end = off + itemsize
    ranges.append((current_start, current_end))
    return ranges


def _make_async_read_fn(reader: _AsyncFsspecReader, meta):
    dtype = SAFETENSORS_DTYPE_MAP[meta["dtype"]]
    shape = meta["shape"]
    base_offset = meta["offset"][0]
    itemsize = dtype.numpy_dtype.itemsize

    async def read(domain: ts.IndexDomain, array: np.ndarray, _params):
        slices = domain.index_exp
        requested_shape = tuple(s.stop - s.start for s in slices)
        buffer = np.empty(requested_shape, dtype=dtype.numpy_dtype)

        # Compute (file_offset, destination_flat_index)
        flat_indices = []
        buffer_strides = np.array([np.prod(requested_shape[i + 1 :], dtype=int) for i in range(len(requested_shape))])

        for idx in np.ndindex(requested_shape):
            global_idx = tuple(s.start + i for s, i in zip(slices, idx))
            flat_index = np.ravel_multi_index(global_idx, shape)
            file_offset = base_offset + flat_index * itemsize
            dest_flat_index = sum(i * s for i, s in zip(idx, buffer_strides))
            flat_indices.append((file_offset, dest_flat_index))

        byte_offsets = [off for off, _ in flat_indices]
        ranges = _coalesce_reads(byte_offsets, itemsize)

        # Fetch coalesced chunks
        chunks = await asyncio.gather(*[reader.read_range(start, end - start) for start, end in ranges])

        # Total buffer as flat view
        flat_buffer = buffer.ravel()

        # Accumulate writes
        all_dest_indices = []
        all_values = []

        for (start, end), chunk in zip(ranges, chunks):
            chunk_array = np.frombuffer(chunk, dtype=dtype.numpy_dtype)
            for off, dest in flat_indices:
                if start <= off < end:
                    rel = (off - start) // itemsize
                    all_dest_indices.append(dest)
                    all_values.append(chunk_array[rel])

        # Vectorized write
        flat_buffer[np.array(all_dest_indices)] = np.array(all_values, dtype=dtype.numpy_dtype)
        array[...] = buffer

    return read


async def load_tensor_dict(gcs_path: str, cache_size=128) -> dict[str, ts.TensorStore]:
    """

    Loads a safetensors file from GCS (or other fsspec file system) into a dictionary of TensorStore objects.
    This allows for efficient sharded reads.

    """
    reader = _AsyncFsspecReader(gcs_path, cache_size=cache_size)
    metadata = await _load_safetensors_metadata(reader)

    tensor_map = {}
    for key, meta in metadata.items():
        read_fn = _make_async_read_fn(reader, meta)
        ts_arr = ts.virtual_chunked(
            read_function=read_fn,
            dtype=SAFETENSORS_DTYPE_MAP[meta["dtype"]],
            shape=meta["shape"],
            loop=asyncio.get_event_loop(),
        )
        tensor_map[key] = ts_arr

    return tensor_map


SAFETENSORS_DTYPE_MAP = {
    "F16": ts.float16,
    "BF16": ts.bfloat16,
    "F32": ts.float32,
    "F64": ts.float64,
    "I8": ts.int8,
    "I16": ts.int16,
    "I32": ts.int32,
    "I64": ts.int64,
    "U8": ts.uint8,
    "U16": ts.uint16,
    "U32": ts.uint32,
    "U64": ts.uint64,
    "BOOL": ts.bool,
}
