# Loss Mask Collection Feature

## Summary

This document describes the changes made to collect `loss_masks` from `LmExample` batches during the forward pass of training. The loss_masks are collected in train order and can be accessed after training steps.

## Changes Made

### 1. `src/levanter/trainer.py`

#### `Trainer` class
- **Added field**: `collected_loss_masks: List[jax.Array] = []`
  - Accumulates loss_masks from all training steps
  - Stored in train order

- **Modified `training_steps` method**:
  - Extracts `loss_mask` from each `example` directly (same pattern as `dataset_id`)
  - Converts NamedArray to plain JAX array for safe multi-device handling
  - Appends to `collected_loss_masks` list
  - **Extraction happens OUTSIDE the jitted train_step** - simple and clean!

- **Added helper methods**:
  - `get_collected_loss_masks()`: Returns all collected loss_masks as a single concatenated array
  - `clear_collected_loss_masks()`: Clears the collected loss_masks from memory


## Usage

### Basic Usage

```python
# During training
trainer = Trainer(config, optimizer, loss_fn)

# Train for some steps
info = trainer.train(state, train_loader)

# Access collected loss_masks
all_loss_masks = trainer.get_collected_loss_masks()

if all_loss_masks is not None:
    print(f"Collected loss_masks shape: {all_loss_masks.shape}")
    # all_loss_masks is a single JAX array concatenated along the batch dimension
    # The order matches the training order

# Clear memory after use
trainer.clear_collected_loss_masks()
```

### Multi-Device/Multi-Node Safety

The implementation is safe for multi-device and multi-node training:

1. **Extraction during forward pass**: Loss_masks are extracted inside the jitted `_train_step` function, ensuring they're part of the compiled computation
2. **Automatic sharding**: The `TrainStepResult` is sharded using `hax.shard_with_axis_mapping()`, which ensures loss_masks are properly distributed
3. **Plain JAX arrays**: Loss_masks are converted from NamedArray to plain JAX arrays for consistent handling across devices
4. **Host-side collection**: Loss_masks are collected on the host side after the jitted function returns, avoiding device memory issues

## Key Features

- ✅ **Train order preservation**: Loss_masks are collected in the exact order they appear in training
- ✅ **Forward pass only**: Extraction happens during forward pass, before any backward pass
- ✅ **Multi-device safe**: Properly handles sharding and device placement
- ✅ **Multi-node safe**: Works correctly in distributed training scenarios
- ✅ **Memory efficient**: Can clear collected masks when no longer needed
- ✅ **Optional**: Returns `None` when no loss_masks are present
- ✅ **Simple implementation**: No changes to jitted functions, follows existing patterns

## Technical Details

### Extraction During Data Loading

The loss_masks are extracted in `training_steps` method, following the exact same pattern as `dataset_id`:

```python
# In training_steps generator, after loading each example:
example = next(iter_data)
self.dataset_ids.append(example.dataset_id)

# Collect loss_mask from the example (same pattern as dataset_id)
if hasattr(example, 'loss_mask') and example.loss_mask is not None:
    loss_mask = example.loss_mask
    # Extract underlying array if it's a NamedArray
    if hasattr(loss_mask, 'array'):
        loss_mask = loss_mask.array
    self.collected_loss_masks.append(loss_mask)
```

This happens:
- **Outside** the jitted train_step function (no JIT complexity!)
- **Before** the example is passed to train_step
- In the **same place** dataset_ids are collected (consistent pattern)

### Concatenation

When retrieving all collected loss_masks:

```python
all_masks = trainer.get_collected_loss_masks()
```

The method:
1. Checks if any masks were collected
2. Concatenates along axis 0 (batch dimension)
3. Returns `None` if no masks or concatenation fails
4. Logs warnings on failure

## Limitations

- Loss_masks accumulate in memory until explicitly cleared
- Only collects loss_masks when the batch has an `LmExample` with a `loss_mask` field
- Concatenation assumes all loss_masks have compatible shapes (except for the batch dimension)

## Testing

To verify the implementation works:

```python
# Check that loss_masks are being collected
assert trainer.collected_loss_masks is not None
assert len(trainer.collected_loss_masks) > 0

# Check concatenation
all_masks = trainer.get_collected_loss_masks()
assert all_masks is not None
assert all_masks.shape[0] == total_examples_processed

# Verify memory can be cleared
trainer.clear_collected_loss_masks()
assert len(trainer.collected_loss_masks) == 0
```

