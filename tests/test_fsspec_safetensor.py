# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from levanter.compat.fsspec_safetensor import SafetensorChunkLoader


@pytest.mark.asyncio
async def test_chunk_loader_roundtrip(tmp_path):
    data = {
        "x": np.random.randn(4, 5).astype(np.float32),
        "y": (np.random.randn(3, 4) * 10).astype(np.float32),
        "z": np.random.randint(0, 255, size=(8,), dtype=np.uint8),
    }
    path = tmp_path / "test.safetensors"
    save_file(data, path)

    reference = load_file(str(path))

    loader = await SafetensorChunkLoader.create(f"file://{path}", chunk_size=256)
    tensors = await loader.read_all()

    for key, value in reference.items():
        np.testing.assert_array_equal(tensors[key], value)


@pytest.mark.asyncio
async def test_dtype_override(tmp_path):
    data = {
        "floaty": np.random.randn(2, 2).astype(np.float32),
        "ints": np.arange(6, dtype=np.int32).reshape(2, 3),
    }
    path = tmp_path / "dtype.safetensors"
    save_file(data, path)

    loader = await SafetensorChunkLoader.create(f"file://{path}", chunk_size=128)
    chunk = loader.chunk_specs[0]
    tensors = await loader.materialize_chunk(chunk, dtype_override=np.float16)

    assert tensors["floaty"].dtype == np.float16
    np.testing.assert_allclose(tensors["floaty"], data["floaty"].astype(np.float16))
    assert tensors["ints"].dtype == np.int32
    np.testing.assert_array_equal(tensors["ints"], data["ints"])


@pytest.mark.asyncio
async def test_materialize_single_tensor(tmp_path):
    data = {
        "a": np.random.randn(16, 16).astype(np.float32),
        "b": np.random.randn(8).astype(np.float32),
    }
    path = tmp_path / "single.safetensors"
    save_file(data, path)

    loader = await SafetensorChunkLoader.create(f"file://{path}", chunk_size=128)
    tensor = await loader.materialize_tensor("b")

    np.testing.assert_array_equal(tensor, data["b"])
