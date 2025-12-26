"""
Example: Collecting and Saving Loss Masks During Training

This example demonstrates how to use the loss_mask collection feature
to collect loss_masks from training batches and save them for later analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Example usage in a training script
def example_training_with_loss_mask_collection(trainer, state, train_loader, output_dir):
    """
    Example training function that collects and saves loss_masks.

    Args:
        trainer: Levanter Trainer instance
        state: TrainerState
        train_loader: Data loader
        output_dir: Directory to save loss_masks
    """

    # Clear any previous collections
    trainer.clear_collected_loss_masks()

    # Train the model
    print("Starting training with loss_mask collection...")
    final_info = trainer.train(state, train_loader)

    # Get all collected loss_masks
    all_loss_masks = trainer.get_collected_loss_masks()

    if all_loss_masks is not None:
        print(f"✓ Collected loss_masks with shape: {all_loss_masks.shape}")

        # Save loss_masks to disk
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to numpy for saving
        loss_masks_np = np.asarray(all_loss_masks)

        # Save as .npy file
        save_path = output_dir / "collected_loss_masks.npy"
        np.save(save_path, loss_masks_np)
        print(f"✓ Saved loss_masks to: {save_path}")

        # Save metadata
        metadata_path = output_dir / "loss_masks_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Shape: {loss_masks_np.shape}\n")
            f.write(f"Dtype: {loss_masks_np.dtype}\n")
            f.write(f"Total elements: {loss_masks_np.size}\n")
            f.write(f"Non-zero elements: {np.count_nonzero(loss_masks_np)}\n")
            f.write(f"Min value: {loss_masks_np.min()}\n")
            f.write(f"Max value: {loss_masks_np.max()}\n")
        print(f"✓ Saved metadata to: {metadata_path}")

        # Clear from memory after saving
        trainer.clear_collected_loss_masks()
        print("✓ Cleared loss_masks from trainer memory")

    else:
        print("⚠ No loss_masks were collected during training")

    return final_info


def example_multi_epoch_collection(trainer, state, train_loader, output_dir, num_epochs=3):
    """
    Example showing loss_mask collection across multiple epochs.

    This demonstrates how to save loss_masks separately for each epoch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # Clear previous epoch's masks
        trainer.clear_collected_loss_masks()

        # Train for one epoch
        # (In practice, you'd need to handle epoch boundaries properly)
        final_info = trainer.train(state, train_loader)
        state = final_info.state

        # Get and save loss_masks for this epoch
        epoch_masks = trainer.get_collected_loss_masks()
        if epoch_masks is not None:
            save_path = output_dir / f"loss_masks_epoch_{epoch:03d}.npy"
            np.save(save_path, np.asarray(epoch_masks))
            print(f"✓ Saved epoch {epoch} loss_masks: {epoch_masks.shape}")

    return state


def example_analysis_of_collected_masks(trainer):
    """
    Example showing how to analyze collected loss_masks.
    """
    all_masks = trainer.get_collected_loss_masks()

    if all_masks is None:
        print("No loss_masks to analyze")
        return

    # Convert to numpy for analysis
    masks_np = np.asarray(all_masks)

    print("\n=== Loss Mask Analysis ===")
    print(f"Total batches collected: {masks_np.shape[0]}")

    # Assuming shape is (batch, sequence_length) or (batch, ...)
    if masks_np.ndim >= 2:
        # Per-example statistics
        per_example_counts = masks_np.sum(axis=tuple(range(1, masks_np.ndim)))
        print(f"Tokens per example - Mean: {per_example_counts.mean():.2f}, "
              f"Std: {per_example_counts.std():.2f}, "
              f"Min: {per_example_counts.min()}, "
              f"Max: {per_example_counts.max()}")

        # Distribution of mask densities
        densities = per_example_counts / np.prod(masks_np.shape[1:])
        print(f"Mask density - Mean: {densities.mean():.3f}, "
              f"Std: {densities.std():.3f}")

        # Find examples with unusual mask patterns
        mean_density = densities.mean()
        std_density = densities.std()
        outliers = np.abs(densities - mean_density) > 2 * std_density
        if outliers.any():
            outlier_indices = np.where(outliers)[0]
            print(f"⚠ Found {len(outlier_indices)} outlier examples (>2σ from mean density)")
            print(f"  Outlier indices: {outlier_indices[:10]}...")  # Show first 10

    print("=" * 50)


# Example integration into a training script
if __name__ == "__main__":
    """
    This is a pseudo-example showing how to integrate loss_mask collection
    into your training script.
    """

    # Your existing training setup
    # config = TrainerConfig(...)
    # optimizer = optax.adam(...)
    # loss_fn = ...
    # trainer = Trainer(config, optimizer, loss_fn)
    # state = load_checkpoint_or_initialize(...)
    # train_loader = trainer.data_loader(dataset)

    # Add loss_mask collection and saving
    # final_info = example_training_with_loss_mask_collection(
    #     trainer, state, train_loader,
    #     output_dir="./outputs/loss_masks"
    # )

    # Or for periodic saving during training:
    # Add a hook to save loss_masks every N steps
    # @trainer.add_hook(every=1000)
    # def save_loss_masks_periodically(info):
    #     masks = trainer.get_collected_loss_masks()
    #     if masks is not None:
    #         np.save(f"./outputs/loss_masks_step_{info.step:06d}.npy", np.asarray(masks))
    #         trainer.clear_collected_loss_masks()

    print("""
    Usage:
    1. Train your model as usual using the Trainer
    2. After training, call trainer.get_collected_loss_masks() to get all masks
    3. Save the masks for later analysis
    4. Call trainer.clear_collected_loss_masks() to free memory

    The loss_masks will be in train order, matching the order of examples
    seen during training.
    """)

