# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from levanter.compat.fsspec_safetensor import load_tensor_dict


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", ["float32", "int8", "uint64"])
async def test_various_dtypes(tmp_path, dtype):
    data = {
        "x": (
            np.random.randint(0, 100, (4, 5)).astype(dtype) if "int" in dtype else np.random.randn(4, 5).astype(dtype)
        ),
    }

    local_path = tmp_path / f"test_{dtype}.safetensors"
    save_file(data, local_path)

    ref = load_file(str(local_path))
    uri = f"file://{local_path}"
    virtual = await load_tensor_dict(uri)

    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])


@pytest.mark.asyncio
async def test_bfloat16_support(tmp_path):
    data = {"bf": np.random.randn(4, 4).astype(np.dtype("bfloat16"))}
    path = tmp_path / "bf16.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")

    np.testing.assert_array_equal(await virtual["bf"].read(), ref["bf"])


@pytest.mark.asyncio
async def test_memory_filesystem():
    import fsspec

    # Create sample file in memory
    data = {"mem": np.random.randn(2, 3).astype(np.float32)}

    with tempfile.NamedTemporaryFile() as f:
        save_file(data, f.name)
        serialized = f.read()

    fs = fsspec.filesystem("memory")
    with fs.open("/test.safetensors", "wb") as f:
        f.write(serialized)

    # Use file URI
    virtual = await load_tensor_dict("memory://test.safetensors")
    ref = data["mem"]
    result = await virtual["mem"].read()
    np.testing.assert_array_equal(result, ref)


@pytest.mark.asyncio
async def test_various_keys_in_one_file(tmp_path):
    data = {
        "x": np.random.randn(4, 5).astype(np.float32),
        "y": np.random.randn(5, 8).astype(np.float32),
        "z": np.random.randn(4, 5).astype(np.int32),
    }

    path = tmp_path / "test.safetensors"
    save_file(data, path)

    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")
    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])
        np.testing.assert_array_equal(await virtual[key].read(), data[key])


@pytest.mark.asyncio
async def test_virtual_slicing(tmp_path):
    data = {"slice": np.arange(100, dtype=np.int32).reshape(10, 10)}

    path = tmp_path / "slice.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")

    # Read a slice
    ts_arr = virtual["slice"]
    sliced = await ts_arr[2:5, 4:7].read()
    expected = ref["slice"][2:5, 4:7]

    np.testing.assert_array_equal(sliced, expected)


# try using gcs
@pytest.mark.asyncio
async def test_gcs(tmp_path):
    import fsspec

    data = {
        "x": np.random.randn(4, 5).astype(np.float32),
        "y": np.random.randn(5, 8).astype(np.float32),
        "z": np.random.randn(4, 5).astype(np.int32),
    }

    local_path = str(tmp_path / "various.safetensors")
    save_file(data, local_path)

    test_data = "gs://levanter-data/test/various.safetensors"

    fs = fsspec.filesystem("gcs")
    try:
        if not fs.exists(test_data):
            fs.put(local_path, test_data)
    except Exception:
        pytest.skip("No test data found")

    virtual = await load_tensor_dict(test_data)
    ref = load_file(local_path)

    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])


@pytest.mark.asyncio
async def test_strided_reads(tmp_path):
    data = {"mat": np.arange(100, dtype=np.float32).reshape(10, 10)}

    path = tmp_path / "weird_strides.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")
    ts_arr = virtual["mat"]

    # Normal read for sanity
    np.testing.assert_array_equal(await ts_arr.read(), ref["mat"])

    # Transpose
    expected = ref["mat"].T
    actual = await ts_arr.transpose([1, 0]).read()
    np.testing.assert_array_equal(actual, expected)

    # Step slicing (every other column)
    expected = ref["mat"][:, ::2]
    actual = await ts_arr[:, ::2].read()
    np.testing.assert_array_equal(actual, expected)

    # Reversed rows
    expected = ref["mat"][::-1]
    actual = await ts_arr[::-1].read()
    np.testing.assert_array_equal(actual, expected)

    # Slice + transpose + step
    expected = ref["mat"][2:8, ::-2].T
    actual = await ts_arr[2:8, ::-2].transpose([1, 0]).read()
    np.testing.assert_array_equal(actual, expected)
