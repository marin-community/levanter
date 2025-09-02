# Predefined Permutation Feature

This feature allows you to permute a dataset using a predefined permutation array loaded from a `.npy` file, instead of generating a random permutation. This is useful when you need reproducible, specific orderings of your data or when you want to use permutations computed by external processes.

## Key Features

- Load permutation from a `.npy` file containing a permutation array
- Preserves original global indices as required for metagradient computations
- Validates that the permutation array is a valid permutation of `range(dataset_length)`
- Seamlessly integrates with existing dataset shuffle functionality

## Usage

### 1. Create a Permutation Array

First, create a permutation array and save it to a `.npy` file:

```python
import numpy as np

# For a dataset of length 1000, create a permutation
dataset_length = 1000
permutation = np.random.permutation(dataset_length)

# Save to file
np.save('/path/to/your/permutation.npy', permutation)
```

### 2. Configure Your Data Config

Add the permutation configuration to your data config:

```yaml
# In your config YAML file
data:
  # ... other data config ...
  shuffle: true
  permutation_type: "predefined"
  permutation_file: "/path/to/your/permutation.npy"
```

### 3. Programmatic Usage

You can also use the feature programmatically:

```python
from levanter.data.permutation import PermutationDataset
import numpy as np

# Load your dataset
dataset = your_dataset  # Any AsyncDataset

# Create permutation dataset from file
perm_dataset = PermutationDataset.from_permutation_file(
    dataset,
    "/path/to/your/permutation.npy"
)

# Or create directly with array
permutation_array = np.load("/path/to/your/permutation.npy")
perm_dataset = PermutationDataset(
    dataset,
    key=None,
    perm_type="predefined",
    permutation_array=permutation_array
)
```

## Behavior

When using a predefined permutation, the dataset behavior is as follows:

**Original dataset**: `[A, B, C, D, ...]` with indices `[0, 1, 2, 3, ...]`

**Permutation array**: `[2, 1, 0, 3, ...]`

**Result**:
- `dataset[0]` returns element `C` with original index `2`
- `dataset[1]` returns element `B` with original index `1`
- `dataset[2]` returns element `A` with original index `0`
- `dataset[3]` returns element `D` with original index `3`

The permutation changes which data elements are returned and preserves their original global indices. This allows you to control the order in which data elements are encountered during training while maintaining the mapping between training examples and their original positions in the dataset, which is crucial for applications like metagradient computation where you need to track which examples correspond to which original data points.

## Validation

The system performs several validations:

1. **Valid permutation**: The array must be a valid permutation of `range(dataset_length)`
2. **Length matching**: The permutation array length must exactly match the dataset length
3. **File existence**: The permutation file must exist and be readable

## Error Handling

Common errors and solutions:

```python
# ValueError: permutation_array must be a valid permutation of range(length)
# Solution: Ensure your array contains each index exactly once

# ValueError: Permutation array length (500) does not match dataset length (1000)
# Solution: Create a permutation array with the correct length

# ValueError: permutation_file must be specified when using predefined permutation type
# Solution: Provide the path to your .npy file in the config

# ValueError: Era shuffling is not supported with predefined permutations
# Solution: Use shuffle: true instead of shuffle: <era_length> with predefined permutations
```

## Configuration Fields

### LMTaskConfig Fields

- `permutation_type`: Set to `"predefined"` to use predefined permutations
- `permutation_file`: Path to the `.npy` file containing the permutation array
- `shuffle`: Set to `true` to enable shuffling (era shuffling not supported with predefined permutations)

## Integration with Existing Code

This feature integrates seamlessly with existing Levanter training scripts:

- **SFT Training**: Works with both single dataset and mixture dataset configurations
- **Pretraining**: Compatible with all existing data loading configurations
- **Evaluation**: Permutation is applied consistently across training and evaluation if configured

## Example Training Command

```bash
python -m levanter.main.train_lm \
  --config_path my_config.yaml \
  --data.permutation_type predefined \
  --data.permutation_file /path/to/permutation.npy \
  --data.shuffle true
```

## Performance Notes

- Loading the permutation array happens once during dataset initialization
- Permutation lookup is O(1) for both single items and batches
- Memory overhead is proportional to dataset size (one int64 per dataset item)
- No computational overhead compared to other permutation types during training