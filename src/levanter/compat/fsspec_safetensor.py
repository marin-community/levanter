# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import struct
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Dict, Iterable, List, Optional, Tuple

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from levanter.utils.jax_utils import broadcast_one_to_all

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


@lru_cache(maxsize=1)
def _default_chunk_size_bytes() -> int:
    """Lazily resolve the default chunk size from the environment."""

    raw = os.environ.get("LEVANTER_GCS_CHUNK_SIZE")
    if not raw:
        return 2 * 1024**3
    try:
        return int(raw)
    except ValueError:
        return 2 * 1024**3


@dataclass(frozen=True)
class _TensorRecord:
    key: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    file_path: str
    byte_start: int
    byte_end: int

    @property
    def byte_length(self) -> int:
        return self.byte_end - self.byte_start


@dataclass(frozen=True)
class _ChunkSpec:
    chunk_id: int
    file_path: str
    byte_start: int
    byte_end: int
    tensors: Tuple[_TensorRecord, ...]

    @property
    def size(self) -> int:
        return self.byte_end - self.byte_start

    def owner(self, process_count: int) -> int:
        if process_count <= 0:
            return 0
        return self.chunk_id % process_count


class _AsyncFsspecReader:
    def __init__(self, path: str, *, cache_size: int = 32):
        protocol, fs_path = fsspec.core.split_protocol(path)
        if protocol is None:
            protocol = "file"
        if protocol == "gs":
            protocol = "gcs"
        self._fs = fsspec.filesystem(protocol, asynchronous=True, anon=False)
        self._path = fs_path
        self._cache_size = cache_size
        self._cache: Dict[Tuple[int, int], bytes] = {}
        self._locks: Dict[Tuple[int, int], asyncio.Lock] = {}

    @property
    def path(self) -> str:
        return self._path

    async def read_range(self, start: int, length: int) -> bytes:
        key = (start, length)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            end = start + length
            if hasattr(self._fs, "_cat_file"):
                data = await self._fs._cat_file(self._path, start=start, end=end)
            else:
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(
                    None,
                    partial(self._fs.cat_file, self._path, start=start, end=end),
                )

            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = data
            self._locks.pop(key, None)
            return data


async def _read_metadata(reader: _AsyncFsspecReader) -> Dict[str, _TensorRecord]:
    header_len_bytes = await reader.read_range(0, 8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    metadata_bytes = await reader.read_range(8, header_len)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    tensors: Dict[str, _TensorRecord] = {}
    data_offset_base = 8 + header_len

    for key, meta in metadata.items():
        if key == "__metadata__":
            continue
        dtype_name: str = meta["dtype"]
        dtype = _SAFETENSOR_DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")

        rel_start, rel_end = meta["data_offsets"]
        tensors[key] = _TensorRecord(
            key=key,
            dtype=dtype,
            shape=tuple(meta["shape"]),
            file_path=reader.path,
            byte_start=data_offset_base + rel_start,
            byte_end=data_offset_base + rel_end,
        )

    return tensors


def _build_chunks(tensors: Iterable[_TensorRecord], chunk_limit: int) -> List[_ChunkSpec]:
    if chunk_limit <= 0:
        raise ValueError("chunk_limit must be positive")

    sorted_tensors = sorted(tensors, key=lambda t: (t.file_path, t.byte_start))
    chunks: List[_ChunkSpec] = []

    current_tensors: List[_TensorRecord] = []
    current_start = 0
    current_end = 0
    current_path = None

    for tensor in sorted_tensors:
        if not current_tensors:
            current_tensors = [tensor]
            current_start = tensor.byte_start
            current_end = tensor.byte_end
            current_path = tensor.file_path
            continue

        next_end = tensor.byte_end
        proposed_size = next_end - current_start
        same_file = tensor.file_path == current_path

        if (not same_file) or (proposed_size > chunk_limit):
            chunk_id = len(chunks)
            chunks.append(
                _ChunkSpec(
                    chunk_id=chunk_id,
                    file_path=current_path if current_path is not None else tensor.file_path,
                    byte_start=current_start,
                    byte_end=current_end,
                    tensors=tuple(current_tensors),
                )
            )
            current_tensors = [tensor]
            current_start = tensor.byte_start
            current_end = tensor.byte_end
            current_path = tensor.file_path
        else:
            current_tensors.append(tensor)
            current_end = max(current_end, tensor.byte_end)

    if current_tensors:
        chunk_id = len(chunks)
        chunks.append(
            _ChunkSpec(
                chunk_id=chunk_id,
                file_path=current_path if current_path is not None else sorted_tensors[-1].file_path,
                byte_start=current_start,
                byte_end=current_end,
                tensors=tuple(current_tensors),
            )
        )

    return chunks


class SafetensorChunkLoader:
    """Chunked safetensors loader that minimises remote round-trips."""

    def __init__(
        self,
        reader: _AsyncFsspecReader,
        chunks: Tuple[_ChunkSpec, ...],
        tensors: Dict[str, _TensorRecord],
    ):
        self._reader = reader
        self._chunks = chunks
        self._tensors = tensors
        self._chunk_by_key = {tensor.key: chunk for chunk in chunks for tensor in chunk.tensors}
        self._chunk_buffers: Dict[int, np.ndarray] = {}
        self._chunk_events: Dict[int, asyncio.Event] = {}
        self._chunk_order: Tuple[_ChunkSpec, ...] = chunks
        self._next_chunk_index: int = 0
        self._order_lock: asyncio.Lock | None = None
        self._order_condition: asyncio.Condition | None = None
        self._process_index = jax.process_index()
        self._process_count = jax.process_count()

    @classmethod
    async def create(
        cls,
        path: str,
        *,
        chunk_size: Optional[int] = None,
    ) -> "SafetensorChunkLoader":
        """Instantiate a loader for `path` with optional chunk sizing override."""

        chunk_limit = chunk_size or _default_chunk_size_bytes()
        reader = _AsyncFsspecReader(path)
        tensors = await _read_metadata(reader)
        chunks = tuple(_build_chunks(tensors.values(), chunk_limit))
        logger.info(
            "Prepared safetensor chunks for %s: %d tensors across %d chunks (max chunk %.2f GiB)",
            path,
            len(tensors),
            len(chunks),
            chunk_limit / 1024**3,
        )
        return cls(reader, chunks, tensors)

    @property
    def chunk_specs(self) -> Tuple[_ChunkSpec, ...]:
        """Return chunk metadata in the order chunks will be materialised."""

        return self._chunks

    @property
    def tensor_records(self) -> Dict[str, _TensorRecord]:
        """Return raw safetensors metadata keyed by tensor name."""

        return self._tensors

    def chunk_for_key(self, key: str) -> _ChunkSpec:
        """Return the chunk metadata for a given tensor key."""

        try:
            return self._chunk_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key} not found in safetensors file") from exc

    async def materialize_chunk(
        self,
        chunk: _ChunkSpec,
        *,
        dtype_override: Optional[jnp.dtype] = None,
    ) -> Dict[str, np.ndarray]:
        """Materialise every tensor in `chunk` as a NumPy array."""

        buffer = await self._get_chunk_buffer(chunk)
        base = chunk.byte_start
        tensors: Dict[str, np.ndarray] = {}
        logger.info(
            "Process %d extracting %d tensors from chunk %d",
            self._process_index,
            len(chunk.tensors),
            chunk.chunk_id,
        )

        for tensor in chunk.tensors:
            start = tensor.byte_start - base
            end = tensor.byte_end - base
            view = buffer[start:end]
            array_view = view.view(tensor.dtype).reshape(tensor.shape)

            if dtype_override is not None and np.issubdtype(array_view.dtype, np.floating):
                target_dtype = np.dtype(dtype_override)
                arrays = array_view.astype(target_dtype, copy=False)
            else:
                arrays = array_view
            tensors[tensor.key] = arrays

        return tensors

    async def materialize_tensor(
        self,
        key: str,
        *,
        dtype_override: Optional[jnp.dtype] = None,
    ) -> np.ndarray:
        """Materialise a single tensor by key as a NumPy array.

        Note that this still materialises the entire chunk that owns the
        tensor, so repeated calls for different tensors in the same chunk are
        cheap but the first call may load more than strictly necessary.
        """

        chunk = self.chunk_for_key(key)
        logger.info(
            "Process %d materialising tensor %s via chunk %d",
            self._process_index,
            key,
            chunk.chunk_id,
        )
        tensors = await self.materialize_chunk(chunk, dtype_override=dtype_override)
        return tensors[key]

    async def read_all(self, *, dtype_override: Optional[jnp.dtype] = None) -> Dict[str, np.ndarray]:
        """Materialise every tensor in the safetensors file."""

        result: Dict[str, np.ndarray] = {}
        for chunk in self._chunks:
            tensors = await self.materialize_chunk(chunk, dtype_override=dtype_override)
            result.update(tensors)
            self.release_chunk(chunk.chunk_id)
        return result

    def release_chunk(self, chunk_id: int) -> None:
        """Drop any cached buffer for the provided chunk id."""

        if chunk_id in self._chunk_buffers:
            logger.info("Process %d released chunk %d", self._process_index, chunk_id)
            self._chunk_buffers.pop(chunk_id, None)
            self._chunk_events.pop(chunk_id, None)

    async def _get_chunk_buffer(self, chunk: _ChunkSpec) -> np.ndarray:
        existing = self._chunk_buffers.get(chunk.chunk_id)
        if existing is not None:
            logger.debug(
                "Process %d reusing cached chunk %d",
                self._process_index,
                chunk.chunk_id,
            )
            return existing

        event = self._chunk_events.get(chunk.chunk_id)
        if event is None:
            event = asyncio.Event()
            self._chunk_events[chunk.chunk_id] = event

        if chunk in self._chunk_order[: self._next_chunk_index]:
            raise RuntimeError(
                f"Chunk {chunk.chunk_id} has already been processed and cannot be materialised again"
            )
        if event.is_set():
            logger.debug(
                "Process %d observed completed chunk %d", self._process_index, chunk.chunk_id
            )
            buffer = self._chunk_buffers.get(chunk.chunk_id)
            if buffer is None:
                raise RuntimeError(
                    f"Chunk {chunk.chunk_id} has been released and cannot be rematerialised"
                )
            return buffer

        condition = self._ensure_condition()
        async with condition:
            while self._next_chunk_index < len(self._chunk_order) and self._chunk_order[self._next_chunk_index] != chunk:
                if event.is_set():
                    return self._chunk_buffers[chunk.chunk_id]
                await condition.wait()

            if event.is_set():
                return self._chunk_buffers[chunk.chunk_id]

            self._next_chunk_index += 1

        is_owner = self._process_index == chunk.owner(self._process_count)
        logger.info(
            "Process %d materialising chunk %d (size %.2f MiB, owner=%d)",
            self._process_index,
            chunk.chunk_id,
            chunk.size / 1024**2,
            chunk.owner(self._process_count),
        )
        if is_owner:
            raw = await self._reader.read_range(chunk.byte_start, chunk.size)
            local_array = np.frombuffer(raw, dtype=np.uint8)
            logger.info(
                "Process %d read chunk %d (%d bytes) from %s",
                self._process_index,
                chunk.chunk_id,
                chunk.size,
                chunk.file_path,
            )
        else:
            local_array = np.empty(chunk.size, dtype=np.uint8)
            local_array.fill(0)

        if self._process_count > 1:
            logger.info(
                "Process %d broadcasting chunk %d buffer (owner=%d)",
                self._process_index,
                chunk.chunk_id,
                chunk.owner(self._process_count),
            )
            local_array = broadcast_one_to_all(local_array, is_source=is_owner)
            multihost_utils.sync_global_devices()

        logger.info(
            "Process %d cached chunk %d (%.2f MiB)",
            self._process_index,
            chunk.chunk_id,
            chunk.size / 1024**2,
        )
        self._chunk_buffers[chunk.chunk_id] = local_array
        event.set()
        condition = self._ensure_condition()
        async with condition:
            condition.notify_all()
        return local_array

    def _ensure_condition(self) -> asyncio.Condition:
        if self._order_condition is None:
            self._order_lock = asyncio.Lock()
            self._order_condition = asyncio.Condition(self._order_lock)
        return self._order_condition


async def create_safetensor_chunk_loader(
    path: str,
    *,
    chunk_size: Optional[int] = None,
) -> SafetensorChunkLoader:
    """Convenience wrapper that forwards to :meth:`SafetensorChunkLoader.create`."""

    return await SafetensorChunkLoader.create(path, chunk_size=chunk_size)
