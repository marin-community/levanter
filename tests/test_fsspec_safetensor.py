# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import jax.numpy as jnp
from fsspec.asyn import AsyncFileSystem
from safetensors.numpy import load_file, save_file
from levanter.compat.fsspec_safetensor import (
    TensorRecord,
    read_safetensors_fsspec,
)
from levanter.compat.hf_checkpoints import HFCheckpointConverter


class _InMemoryAsyncFS(AsyncFileSystem):
    def __init__(self, payload: bytes):
        super().__init__()
        self._payload = payload

    async def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **_) -> bytes:
        start = 0 if start is None else start
        end = len(self._payload) if end is None else end
        return self._payload[start:end]


@pytest.mark.asyncio
async def test_async_tensor_record_get_slice_handles_strides():
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    payload = array.tobytes()
    fs = _InMemoryAsyncFS(payload)
    record = TensorRecord(
        key="foo",
        dtype=array.dtype,
        shape=array.shape,
        file_path="mem://foo",
        byte_start=0,
        byte_end=len(payload),
        fs=fs,
    )

    result = await record.get_slice((slice(1, 3), slice(2, 5)))
    np.testing.assert_array_equal(result, array[1:3, 2:5])

    stepped = await record.get_slice((Ellipsis, slice(None, None, 2)))
    np.testing.assert_array_equal(stepped, array[:, ::2])

    row_stepped = await record.get_slice((slice(None, None, 2), 3))
    np.testing.assert_array_equal(row_stepped, array[::2, 3])

    zero = await record.get_slice((slice(0, 0), slice(None)))
    assert zero.shape == (0, array.shape[1])

    with pytest.raises(NotImplementedError):
        await record.get_slice((slice(None, None, -1), slice(None)))


@pytest.mark.asyncio
async def test_read_safetensors_fsspec_roundtrip(tmp_path):
    data = {
        "foo": np.arange(12, dtype=np.float32).reshape(3, 4),
        "bar": (np.random.randn(2, 3) * 5).astype(np.float32),
        "baz": np.arange(6, dtype=np.int32),
    }
    path = tmp_path / "roundtrip.safetensors"
    save_file(data, path)

    arrays = await read_safetensors_fsspec(
        f"file://{path}",
        dtype_override=jnp.float16,
        sharding_fn=lambda _: None,
    )

    assert set(arrays.keys()) == set(data.keys())
    np.testing.assert_array_equal(np.asarray(arrays["baz"]), data["baz"])
    assert arrays["foo"].dtype == jnp.float16
    np.testing.assert_allclose(np.asarray(arrays["foo"]), data["foo"].astype(np.float16))


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


@pytest.mark.asyncio
async def test_dtype_override(tmp_path):
    data = {
        "floaty": np.random.randn(2, 2).astype(np.float32),
        "ints": np.arange(6, dtype=np.int32).reshape(2, 3),
    }
    path = tmp_path / "dtype.safetensors"
    save_file(data, path)

    tensors = await read_safetensors_fsspec(
        f"file://{path}",
        dtype_override=jnp.bfloat16,
        sharding_fn=lambda _: None,
    )

    assert tensors["floaty"].dtype == jnp.bfloat16
    np.testing.assert_equal(tensors["floaty"], data["floaty"].astype(jnp.bfloat16))
    assert tensors["ints"].dtype == np.int32
    np.testing.assert_array_equal(tensors["ints"], data["ints"])
