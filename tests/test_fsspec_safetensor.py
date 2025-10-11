# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import fsspec
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from levanter.compat.fsspec_safetensor import SafetensorChunkLoader
from levanter.compat.hf_checkpoints import HFCheckpointConverter


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


@pytest.mark.asyncio
async def test_out_of_order_chunk_access(tmp_path):
    # Force multiple chunks by setting a small chunk size relative to tensor payloads.
    data = {
        "first": np.random.randn(128).astype(np.float32),
        "second": np.random.randn(64).astype(np.float32),
    }
    path = tmp_path / "out_of_order.safetensors"
    save_file(data, path)

    loader = await SafetensorChunkLoader.create(f"file://{path}", chunk_size=256)

    second = await asyncio.wait_for(loader.materialize_tensor("second"), timeout=1.0)
    np.testing.assert_array_equal(second, data["second"])

    first = await asyncio.wait_for(loader.materialize_tensor("first"), timeout=1.0)
    np.testing.assert_array_equal(first, data["first"])


@pytest.mark.asyncio
async def test_chunk_loader_with_custom_fs(tmp_path):
    data = {
        "foo": np.random.randn(4, 4).astype(np.float32),
        "bar": np.random.randn(4, 4).astype(np.float32),
    }
    path = tmp_path / "custom_fs.safetensors"
    save_file(data, path)

    fs = fsspec.filesystem("file")
    loader = await SafetensorChunkLoader.create(str(path), chunk_size=128, fs=fs)

    tensors = await loader.read_all()
    np.testing.assert_array_equal(tensors["foo"], data["foo"])
    np.testing.assert_array_equal(tensors["bar"], data["bar"])


def test_load_from_remote_file_url(tmp_path, monkeypatch):
    data = {
        "foo": np.random.randn(4, 4).astype(np.float32),
        "bar": np.random.randn(3, 2).astype(np.float32),
    }
    path = tmp_path / "model.safetensors"
    save_file(data, path)

    expected = load_file(str(path))

    monkeypatch.setattr("levanter.compat.hf_checkpoints.best_effort_sharding", lambda shape: None)

    def _jit_stub(fn, *args, **kwargs):
        def _wrapped(x):
            return fn(x)

        return _wrapped

    monkeypatch.setattr("levanter.compat.hf_checkpoints.jax.jit", _jit_stub)
    monkeypatch.setattr("levanter.compat.hf_checkpoints.jax.lax.with_sharding_constraint", lambda x, _: x)

    converter = HFCheckpointConverter.__new__(HFCheckpointConverter)
    converter.__dict__.update(
        {
            "LevConfigClass": None,
            "reference_checkpoint": None,
            "HfConfigClass": None,
            "tokenizer": None,
            "feature_extractor": None,
            "config_overrides": None,
            "trust_remote_code": False,
            "ignore_prefix": None,
        }
    )

    remote_state = converter._load_from_remote(f"file://{tmp_path}", dtype=None)

    assert set(remote_state.keys()) == set(expected.keys())
    for key in expected:
        np.testing.assert_array_equal(np.array(remote_state[key]), expected[key])
