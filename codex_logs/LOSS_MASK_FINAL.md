# Loss Mask Collection - Final Implementation

## Overview

Loss masks are collected as a **1D indicator array** in sequential train order, where each entry indicates whether that example had any non-zero loss mask.

## Why 1D?

Since loss masks are typically binary per example (either the example contributes to loss or not), we only need to store a single bit per example rather than the full `(batch_size, seq_len)` mask. This saves significant storage:

- **Old approach**: `(num_examples, seq_len)` - full 2D masks
- **New approach**: `(num_examples,)` - 1D boolean indicators
- **Savings**: ~1000x smaller for seq_len=1024!

## Implementation

### 1. Initialize 1D Array

In `train_lm.py`:

```python
# Initialize global loss_masks indicator array (1D boolean array)
total_examples = trainer.config.num_train_steps * trainer.config.train_batch_size
loss_masks_global = jnp.zeros(total_examples, dtype=jnp.int32)
```

**Shape**: `(total_examples,)` - one entry per training example

### 2. Fill During Training

In `trainer.py`, during each training step:

```python
# Extract loss_mask from example
loss_mask = example.loss_mask.array  # Shape: (batch_size, seq_len)

# Compute indicator: 1 if row sum > 0, else 0
mask_indicators = (loss_mask.sum(axis=-1) > 0).astype(jnp.int32)  # Shape: (batch_size,)

# Calculate sequential indices based on step
batch_size = loss_mask.shape[0]
start_idx = int(state.step) * batch_size
end_idx = start_idx + batch_size

# Update global array
self.loss_masks_global = self.loss_masks_global.at[start_idx:end_idx].set(mask_indicators)
```

**Key points**:
- Sum each row of the loss_mask: `loss_mask.sum(axis=-1)`
- Check if sum > 0: `(... > 0).astype(jnp.int32)`
- Store in sequential order: first batch → indices 0-7, second batch → indices 8-15, etc.

### 3. Save as Single File

In `train_lm.py`:

```python
_save_array_tpu_safe(out_dir, 'loss_masks_global.npy', loss_masks_global, gather_to_single_file=True)
```

With `gather_to_single_file=True`:
- Gathers the distributed array from all hosts
- Rank 0 saves a single file: `loss_masks_global.npy`
- No multiple shard files (`.r0`, `.r1`, etc.)

## Result

**Saved file**: `loss_masks_global.npy`

**Contents**:
- Shape: `(num_train_steps * train_batch_size,)`
- Dtype: `int32`
- Values: `0` or `1`
  - `1` = example had non-zero loss mask (contributed to training)
  - `0` = example had zero loss mask (excluded from training)
- Order: Sequential train order (first example seen → index 0, second → index 1, etc.)

## Usage

```python
import numpy as np

# Load the indicator array
loss_masks = np.load("loss_masks_global.npy")

print(f"Total examples: {len(loss_masks)}")
print(f"Active examples: {loss_masks.sum()}")
print(f"Inactive examples: {(loss_masks == 0).sum()}")
print(f"Fraction active: {loss_masks.mean():.1%}")

# Get indices of active examples
active_indices = np.where(loss_masks == 1)[0]
inactive_indices = np.where(loss_masks == 0)[0]

# Match with data_weight_vector
data_weights = np.load("data_weight_vector.npy")
assert len(loss_masks) == len(data_weights)

# Examples with non-zero mask AND high weight
high_value_examples = np.where((loss_masks == 1) & (data_weights > 0.5))[0]
```

## Comparison with data_weight_vector

Both arrays have the same shape and indexing:

| Array | Shape | Values | Meaning |
|-------|-------|--------|---------|
| `data_weight_vector` | `(N,)` | Float (0.0 to 1.0) | Weight for meta-gradient |
| `loss_masks_global` | `(N,)` | Int (0 or 1) | Whether example contributed to loss |

Both indexed by **sequential train order**.

## Advantages

✅ **Compact**: ~1000x smaller than storing full masks
✅ **Sequential order**: Matches training order exactly
✅ **Single file**: Easy to load and use
✅ **Consistent API**: Same pattern as `data_weight_vector`
✅ **Distributed safe**: Works correctly on multi-host TPU/GPU

## Example Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Load arrays
data_weights = np.load("data_weight_vector.npy")
loss_masks = np.load("loss_masks_global.npy")

# Plot training dynamics
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Plot 1: Loss mask over training
axes[0].plot(loss_masks)
axes[0].set_ylabel("Loss Mask Active")
axes[0].set_xlabel("Training Example Index")
axes[0].set_title("Loss Mask Activity During Training")

# Plot 2: Combined view
axes[1].scatter(range(len(loss_masks)), data_weights,
                c=loss_masks, cmap='RdYlGn', alpha=0.5)
axes[1].set_ylabel("Data Weight")
axes[1].set_xlabel("Training Example Index")
axes[1].set_title("Data Weights colored by Loss Mask")
plt.tight_layout()
plt.savefig("training_analysis.png")

# Statistics
print(f"Examples with mask=1, weight>0.5: {((loss_masks==1) & (data_weights>0.5)).sum()}")
print(f"Examples with mask=0, weight>0.5: {((loss_masks==0) & (data_weights>0.5)).sum()}")
print(f"Examples with mask=1, weight<0.5: {((loss_masks==1) & (data_weights<0.5)).sum()}")
```

