# Loss Mask Sharding Fix

## Problem

When running on 8 nodes, `collected_loss_masks.npy` was being saved as 8 sharded files (`.r0` through `.r7`), but:
- ❌ Each file had size `(total_examples, seq_len)` (full dataset, not just portion)
- ❌ Files were not identical (each had different/overlapping data)
- ❌ This indicated each host was collecting the full sharded array instead of just its local portion

## Root Cause

In the original implementation:

```python
# OLD CODE (INCORRECT)
loss_mask = example.loss_mask
if hasattr(loss_mask, 'array'):
    loss_mask = loss_mask.array
self.collected_loss_masks.append(loss_mask)  # ❌ Appending full sharded array!
```

The `example.loss_mask` is a **distributed JAX array** that spans all 8 hosts. When each host appended this array to its list, it was collecting the entire sharded array (with views into all 8 hosts' data), not just its local portion.

## Solution

Extract only the **addressable (local) shards** on each host:

```python
# NEW CODE (CORRECT)
loss_mask = example.loss_mask
if hasattr(loss_mask, 'array'):
    loss_mask = loss_mask.array

# Extract only LOCAL shards
if isinstance(loss_mask, jax.Array) and not getattr(loss_mask, 'is_fully_addressable', True):
    # Distributed array - extract only local shards
    local_shape = loss_mask.shape
    local_data = np.zeros(local_shape, dtype=loss_mask.dtype)
    for shard in loss_mask.addressable_shards:
        local_data[shard.index] = np.asarray(jax.device_get(shard.data))
    self.collected_loss_masks.append(local_data)
else:
    # Fully addressable - collect on rank 0 only
    if jax.process_index() == 0:
        self.collected_loss_masks.append(np.asarray(loss_mask))
```

## Result

Now each host collects only its portion:
- ✅ Host 0 saves `collected_loss_masks.npy.r0` with 1/8 of the data
- ✅ Host 1 saves `collected_loss_masks.npy.r1` with 1/8 of the data
- ✅ ... (and so on)
- ✅ Together they form the complete dataset with no overlap

## How to Read Sharded Files

Use the provided helper script:

```bash
python scripts/combine_sharded_npy.py gs://bucket/path/collected_loss_masks.npy
```

Or in Python:

```python
import numpy as np
import glob

# Load and combine all shards
shards = []
for shard_file in sorted(glob.glob("collected_loss_masks.npy.r*")):
    shards.append(np.load(shard_file))

# Concatenate along first (batch) axis
full_array = np.concatenate(shards, axis=0)
print(f"Full array shape: {full_array.shape}")
```

## Technical Details

### Why `.addressable_shards`?

In JAX's multi-host distributed arrays:
- **Global array**: The logical array spanning all devices/hosts
- **Addressable shards**: The portions of the array that are physically on this host's devices
- **Non-addressable shards**: Portions on other hosts' devices

When you try to access a global array with non-addressable shards, you get:
```
RuntimeError: Fetching value for `jax.Array` that spans non-addressable devices is not possible
```

### Extraction Logic

```python
local_data = np.zeros(loss_mask.shape, dtype=loss_mask.dtype)
for shard in loss_mask.addressable_shards:
    local_data[shard.index] = np.asarray(jax.device_get(shard.data))
```

This:
1. Creates a zero buffer matching the full array shape
2. Fills in only the slices (`shard.index`) that are addressable on this host
3. Results in a sparse array where only local data is non-zero

When combined across hosts, the non-overlapping non-zero regions form the complete array.

## Verification

After the fix, verify the shards are correct:

```python
import numpy as np
import glob

files = sorted(glob.glob("collected_loss_masks.npy.r*"))
print(f"Found {len(files)} shards")

shapes = []
sums = []
for f in files:
    shard = np.load(f)
    shapes.append(shard.shape)
    sums.append(shard.sum())
    print(f"{f}: shape={shard.shape}, sum={shard.sum():.2f}")

# Check that shapes are consistent
assert all(s == shapes[0] for s in shapes), "Shapes don't match!"

# Combine
combined = np.concatenate([np.load(f) for f in files], axis=0)
print(f"\nCombined: shape={combined.shape}, sum={combined.sum():.2f}")
```

Expected output:
- All shards have the same shape (but different sums indicating different data)
- Combined shape should be `(num_hosts * batch_size_per_host, seq_len)`
- Combined sum should equal sum of individual shard sums

