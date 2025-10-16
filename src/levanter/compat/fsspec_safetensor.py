# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from fsspec import AbstractFileSystem

from levanter.utils.jax_utils import broadcast_one_to_all, sync_global_devices

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


DEFAUlT_CHUNK_SIZE = 1 * 1024**3


@lru_cache(maxsize=1)
def _default_chunk_size_bytes() -> int:
    """Lazily resolve the default chunk size from the environment."""

    raw = os.environ.get("LEVANTER_GCS_CHUNK_SIZE")
    if not raw:
        return DEFAUlT_CHUNK_SIZE

    return int(raw)


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


# https://huggingface.co/docs/safetensors/en/index#format
def _read_metadata(fs: AbstractFileSystem, path) -> Dict[str, _TensorRecord]:
    header_len_bytes = fs.cat_file(path, start=0, end=8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    metadata_bytes = fs.cat_file(path, start=8, end=8 + header_len)
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
            file_path=path,
            byte_start=data_offset_base + rel_start,
            byte_end=data_offset_base + rel_end,
        )

    return tensors


def _build_chunks(tensors: Iterable[_TensorRecord], chunk_limit: int) -> List[_ChunkSpec]:
    """
    Group tensors into chunks that are at most `chunk_limit` bytes in size. If a single
    tensor exceeds `chunk_limit`, it will be placed in its own chunk.
    """
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
    """Chunked safetensors loader that avoids downloading the entire file at once."""

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        chunks: Tuple[_ChunkSpec, ...],
        tensors: Dict[str, _TensorRecord],
    ):
        self._fs = fs
        self._path = path
        self._chunks = chunks
        self._tensors = tensors
        self._chunk_by_key = {tensor.key: chunk for chunk in chunks for tensor in chunk.tensors}
        self._chunk_buffers: Dict[int, np.ndarray] = {}
        self._process_index = jax.process_index()
        self._process_count = jax.process_count()
        self._prefetch_lock = threading.Lock()
        self._prefetch_complete = False
        self._shared_chunks: Set[int] = set()
        self._owned_chunks: Tuple[_ChunkSpec, ...] = tuple(
            chunk for chunk in chunks if chunk.owner(self._process_count) == self._process_index
        )
        self._ensure_owned_chunks_prefetched()

    @classmethod
    def create(
        cls,
        path: str,
        *,
        chunk_size: Optional[int] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> "SafetensorChunkLoader":
        """Instantiate a loader for `path` with optional chunk sizing override."""
        protocol, fs_path = fsspec.core.split_protocol(path)
        if protocol is None:
            protocol = "file"

        if fs is None:
            fs = fsspec.filesystem(protocol, asynchronous=True, anon=False)

        chunk_limit = chunk_size or _default_chunk_size_bytes()
        tensors = _read_metadata(fs, path)
        chunks = tuple(_build_chunks(tensors.values(), chunk_limit))
        maximum_chunk_size = max(chunk.size for chunk in chunks) if chunks else 0
        logger.info(
            "Prepared safetensor chunks for %s: %d tensors across %d chunks (maximum %.2f MiB)",
            path,
            len(tensors),
            len(chunks),
            maximum_chunk_size / 1024**2,
        )
        return cls(fs, path, chunks, tensors)

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

    def materialize_chunk(
        self,
        chunk: _ChunkSpec,
        *,
        dtype_override: Optional[jnp.dtype] = None,
    ) -> Dict[str, np.ndarray]:
        """Materialise every tensor in `chunk` as a NumPy array."""

        buffer = self._get_chunk_buffer(chunk)
        base = chunk.byte_start
        tensors: Dict[str, np.ndarray] = {}
        logger.debug(
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

    def materialize_tensor(
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
        logger.debug(
            "Process %d materialising tensor %s via chunk %d",
            self._process_index,
            key,
            chunk.chunk_id,
        )
        tensors = self.materialize_chunk(chunk, dtype_override=dtype_override)
        return tensors[key]

    def read_all(self, *, dtype_override: Optional[jnp.dtype] = None) -> Dict[str, np.ndarray]:
        """Materialise every tensor in the safetensors file."""

        result: Dict[str, np.ndarray] = {}
        for chunk in self._chunks:
            tensors = self.materialize_chunk(chunk, dtype_override=dtype_override)
            result.update(tensors)
            self.release_chunk(chunk.chunk_id)
        return result

    def release_chunk(self, chunk_id: int) -> None:
        """Drop any cached buffer for the provided chunk id."""

        if chunk_id in self._chunk_buffers:
            logger.debug("Process %d released chunk %d", self._process_index, chunk_id)
            self._chunk_buffers.pop(chunk_id, None)
            self._shared_chunks.discard(chunk_id)

    def _get_chunk_buffer(self, chunk: _ChunkSpec) -> np.ndarray:
        self._ensure_owned_chunks_prefetched()
        sync_global_devices(repr(chunk))
        time_in = time.time()
        is_owner = self._process_index == chunk.owner(self._process_count)
        existing = self._chunk_buffers.get(chunk.chunk_id)
        if existing is not None and (not is_owner or chunk.chunk_id in self._shared_chunks):
            logger.debug(
                "Process %d reusing chunk %d",
                self._process_index,
                chunk.chunk_id,
            )
            return existing

        if existing is not None:
            local_array = existing
        else:
            logger.info(
                "Process %d materialising chunk %d (size %.2f MiB, owner=%d)",
                self._process_index,
                chunk.chunk_id,
                chunk.size / 1024**2,
                chunk.owner(self._process_count),
            )
            if is_owner:
                raw = self._read_chunk_bytes(chunk)
                local_array = np.frombuffer(raw, dtype=np.uint8)
                logger.info(
                    "Process %d read chunk %d (%d bytes) from %s",
                    self._process_index,
                    chunk.chunk_id,
                    chunk.size,
                    chunk.file_path,
                )
            else:
                local_array = np.zeros(chunk.size, dtype=np.uint8)

        if self._process_count > 1:
            logger.info(
                "Process %d broadcasting chunk %d buffer (owner=%d)",
                self._process_index,
                chunk.chunk_id,
                chunk.owner(self._process_count),
            )
            broadcast_in = time.time()
            local_array = broadcast_one_to_all(local_array, is_source=is_owner)
            broadcast_out = time.time()
            logger.info(
                "Process %d broadcasted chunk %d in %.2f seconds",
                self._process_index,
                chunk.chunk_id,
                broadcast_out - broadcast_in,
            )

        if existing is None:
            time_end = time.time()
            logger.info(
                "Process %d cached chunk %d (%.2f MiB) in %.2f seconds (%.2f MiB/s)",
                self._process_index,
                chunk.chunk_id,
                chunk.size / 1024**2,
                time_end - time_in,
                (chunk.size / 1024**2) / (time_end - time_in),
            )
        self._chunk_buffers[chunk.chunk_id] = local_array
        self._shared_chunks.add(chunk.chunk_id)
        return local_array

    def _read_chunk_bytes(self, chunk: _ChunkSpec) -> bytes | bytearray:
        return self._fs.cat_file(self._path, start=chunk.byte_start, end=chunk.byte_end)

    def _ensure_owned_chunks_prefetched(self) -> None:
        if self._prefetch_complete:
            return

        with self._prefetch_lock:
            if self._prefetch_complete:
                return

            prefetch_start = time.time()
            prefetched_count = 0
            prefetched_bytes = 0
            for chunk in self._owned_chunks:
                if chunk.chunk_id in self._chunk_buffers:
                    continue
                raw = self._read_chunk_bytes(chunk)
                self._chunk_buffers[chunk.chunk_id] = np.frombuffer(raw, dtype=np.uint8)
                prefetched_count += 1
                prefetched_bytes += chunk.size

            prefetch_end = time.time()

            if prefetched_count > 0:
                logger.info(
                    "Process %d prefetched %d owned chunk(s) totalling %.2f MiB in %.2f seconds (%.2f MiB/s)",
                    self._process_index,
                    prefetched_count,
                    prefetched_bytes / 1024**2,
                    prefetch_end - prefetch_start,
                    (prefetched_bytes / 1024**2) / (prefetch_end - prefetch_start),
                )
            self._prefetch_complete = True
