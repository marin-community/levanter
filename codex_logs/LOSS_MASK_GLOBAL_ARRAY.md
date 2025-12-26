# Loss Mask Collection - Global Array Approach

## Overview

Loss masks are now collected using a **global array indexed by train order**, matching the pattern used for `data_weight_vector`. This ensures the loss masks maintain the exact training order and can be easily saved/loaded in distributed training.

## Implementation

### 1. Initialize Global Array

In `train_lm.py`, a global loss_masks array is pre-allocated:

```python
# Initialize global loss_masks array (same pattern as data_weight_vector)
seq_len = state.model.Pos.size
total_examples = trainer.config.num_train_steps * trainer.config.train_batch_size
loss_masks_global = jnp.zeros((total_examples, seq_len), dtype=jnp.int32)
```

**Shape**: `(num_train_steps * train_batch_size, seq_len)`
**Indexing**: Global train order (same as `data_weight_vector`)

### 2. Pass to Trainer

The array is passed to `train_and_replay` and then to `train()`:

```python
ret = trainer.train_and_replay(
    state, train_loader, reversed_train_loader, reward_loader,
    data_weight_vector, segment_starts,
    train_only=config.train_only,
    loss_masks_global=loss_masks_global,  # ← Pass the global array
)
```

### 3. Fill During Training

In `training_steps()`, loss masks are filled at the appropriate indices:

```python
# Get the global indices for this batch
indices = example.index  # Global train-order indices

# Extract loss_mask from example
loss_mask = example.loss_mask
if hasattr(loss_mask, 'array'):
    loss_mask = loss_mask.array

# Update the global array at these indices
self.loss_masks_global = self.loss_masks_global.at[indices].set(loss_mask)
```

**Key points**:
- `example.index` contains the global train-order indices for this batch
- JAX's `.at[indices].set()` creates a new array with updated values
- This works correctly in distributed training - each host updates its local shards

### 4. Return and Save

The filled array is returned and saved:

```python
# Return from train_and_replay
return reward, metagrads, dataset_ids_global, local_indices_global, self.loss_masks_global

# Save (in train_lm.py)
_save_array_tpu_safe(out_dir, 'loss_masks_global.npy', loss_masks_global)
```

## Advantages Over Previous Approach

### ❌ Old Approach (List-based)
- Collected loss_masks in a list per step
- Concatenated at the end
- Resulted in host-by-host order, not train order
- Required complex shard extraction logic
- Files had overlapping/duplicate data

### ✅ New Approach (Global Array)
- Pre-allocated array indexed by train order
- Filled in-place during training
- **Train order automatically preserved** by using global indices
- Matches `data_weight_vector` pattern (consistent API)
- Sharding handled correctly by JAX
- Single source of truth

## Train Order Preservation

The key insight is that **train order is encoded in the indices**, not the file structure:

```
Step 0: Batch indices [0, 1, 2, 3, 4, 5, 6, 7]        (split across 8 hosts)
Step 1: Batch indices [8, 9, 10, 11, 12, 13, 14, 15]  (split across 8 hosts)
Step 2: Batch indices [16, 17, 18, 19, 20, 21, 22, 23] (split across 8 hosts)
...
```

Each host fills in its portion:
- Host 0 fills indices: 0, 8, 16, 24, ...
- Host 1 fills indices: 1, 9, 17, 25, ...
- Host 2 fills indices: 2, 10, 18, 26, ...
- etc.

When saved as sharded files and loaded, the global array maintains the correct train order because the indices are used during filling.

## Usage

### Accessing Loss Masks

```python
# Get the global array
loss_masks = trainer.get_collected_loss_masks()

# Shape: (num_train_steps * train_batch_size, seq_len)
# Index by global train order (matching data_weight_vector)
```

### Loading Saved Files

```python
import numpy as np
import glob

# Load sharded files
shards = [np.load(f) for f in sorted(glob.glob("loss_masks_global.npy.r*"))]

# Combine (if needed, though often you can work with individual shards)
full_array = np.concatenate(shards, axis=0)

# Index by global train order
example_100_mask = full_array[100]  # Loss mask for training example #100
```

### Matching with Data Weights

Since both use the same indexing scheme:

```python
# Load both arrays
data_weights = np.load("data_weight_vector.npy")
loss_masks = np.load("loss_masks_global.npy") # or combined from shards

# They match in order!
assert len(data_weights) == len(loss_masks)

# Example: Get high-weight examples with specific mask patterns
high_weight_indices = np.where(data_weights > 0.5)[0]
high_weight_masks = loss_masks[high_weight_indices]
```

## Distributed Training

In multi-host training:
1. `loss_masks_global` is a **sharded JAX array** distributed across hosts
2. Each host fills its local shards using `.at[indices].set()`
3. `_save_array_tpu_safe` automatically handles the sharding:
   - Detects non-fully-addressable arrays
   - Each host saves its local shards to a separate file (`.r0`, `.r1`, etc.)
4. The combination of all shard files gives the complete array **in train order**

## API Summary

### Trainer Methods

```python
# Get the global loss_masks array
loss_masks = trainer.get_collected_loss_masks()
# Returns: jnp.ndarray of shape (total_examples, seq_len) or None

# Clear from memory
trainer.clear_collected_loss_masks()
```

### train_and_replay Signature

```python
def train_and_replay(
    self, state, train_loader, reversed_train_loader, val_loader,
    data_weight_vector, segment_starts,
    train_only=False,
    loss_masks_global: Optional[jnp.ndarray] = None  # ← New parameter
) -> tuple[reward, metagrads, dataset_ids, local_indices, loss_masks_global]:
    ...
```

Returns the filled `loss_masks_global` array as the 5th element.

