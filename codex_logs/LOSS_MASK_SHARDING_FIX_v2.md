# Loss Mask Collection - Multi-Node Sharding Fix

## Problem

When running on multi-node setups (e.g., 8 nodes), the `loss_masks_global` array had regions of zeros even though training proceeded normally.

### Root Cause

The batch data in JAX distributed training is **sharded** across processes:
- All processes see the same global batch size (e.g., 1024)
- Each process holds only a local shard (e.g., 128 per process for 8 processes)
- `loss_mask.shape[0]` returns the **global** batch size, not the local shard size

The `loss_masks_global` array was initialized as a **replicated** array:
- Each process had its own full copy of the array
- When writing `mask_indicators` (sharded) into `loss_masks_global` (replicated):
  - Each process only wrote its local shard
  - Only that process's copy got updated
  - Other indices remained zero

## Solution

Create `loss_masks_global` as a **sharded array** matching the batch data sharding:

```python
# Create sharded array along batch dimension
from jax.sharding import NamedSharding, PartitionSpec

mesh = trainer.device_mesh
sharding = NamedSharding(mesh, PartitionSpec(trainer.config.data_axis_name))
loss_masks_global = jax.device_put(jnp.zeros(total_examples, dtype=jnp.int32), sharding)
```

When both source (`mask_indicators`) and destination (`loss_masks_global`) are sharded the same way:
- Each process writes to its corresponding shard
- JAX automatically handles the distribution
- No gaps or overwrites

## Changes Made

### `src/levanter/main/train_lm.py`

1. **Modified initialization** (lines ~524-543):
   - Create `loss_masks_global` as a sharded array using `NamedSharding`
   - Shard along the data axis (typically the batch dimension)
   - Log the sharding info for debugging

### `src/levanter/trainer.py`

1. **Simplified indexing** (lines ~1202-1207):
   - Use simple `step * batch_size` indexing (reverted complex per-process indexing)
   - Added comment explaining that sharding handles distribution automatically

2. **Added diagnostic logging** (lines ~1214-1227):
   - Warn if all mask indicators are zero
   - Warn if loss_mask attribute is missing
   - Log every 100 steps for normal operation

## Testing

Run training on multi-node and check:

1. **Initialization log**:
   ```
   [LossMask] Initialized global loss_masks indicator array with shape: (N,), sharding: ...
   ```

2. **Collection logs** (every 100 steps):
   ```
   [LossMask] Step 0: Filled loss_mask indicators at [0:1024], active: 800/1024
   [LossMask] Step 100: Filled loss_mask indicators at [102400:103424], active: 750/1024
   ```

3. **No zero regions**: After training, verify `loss_masks_global` has no unexpected zero regions

## Key Insights

- JAX arrays can be **replicated** (all processes have full copy) or **sharded** (distributed across processes)
- When mixing replicated and sharded arrays in operations, only local data is affected
- Match the sharding of source and destination arrays for correct distributed updates
- Use `jax.device_put(array, sharding)` to create arrays with explicit sharding

