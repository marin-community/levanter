# Unified checkpoint and initialization strategy

## Context
- `train_lm` currently supports multiple overlapping initialization options:
  - start from scratch when no checkpoint is found
  - resume from the latest checkpoint of the current run
  - load a checkpoint from another run via `trainer.load_checkpoint_path`
  - load only model weights via `trainer.initialize_from`
  - load weights and optimizer state while resetting data via `initialize_from_checkpoint_path`
  - bootstrap from HuggingFace checkpoints using `--initialize_from_hf`
- This multiplicity is confusing and requires different code paths.

One objective is to store the model configuration inside Levanter checkpoints to make reuse safer.

## Goals
1. Provide a single, coherent API for all initialization cases.
2. Persist model configuration (and optionally the full `TrainLmConfig`) in every checkpoint.
3. Allow reinitializing training data independently from model/optimizer loading.
4. Maintain backward compatibility for existing flags while providing a migration path.

## Proposed design
- Introduce an `InitConfig` dataclass with variants:
  - `scratch` – default; start from randomly initialized model.
  - `checkpoint(path, load_optimizer=True, reset_dataloader=False)` – load from Levanter checkpoint. Can optionally skip optimizer state or reset data position.
  - `huggingface(repo_or_path, use_hf_config=False)` – load weights from HF checkpoint. Optionally replace model config with HF config.
- `train_lm` and `Trainer` accept an `init` field instead of multiple flags. Deprecated flags (`initialize_from_hf`, `initialize_from`, `initialize_from_checkpoint_path`, `load_checkpoint_path`) are mapped to this structure for compatibility.
- A single initialization function resolves `InitConfig` and returns the starting `TrainerState` along with an indicator whether the data loader should resume or reset.

## Checkpoint changes
- Extend `save_checkpoint` to include a serialized `model_config` (and ideally the full `TrainLmConfig`).
- `load_checkpoint` returns both the state and saved config, enabling validation that the current run's config matches (or deliberate overrides).
- Implement versioning in checkpoint metadata to ease future migrations.

## Step-by-step plan
- [ ] Design `InitConfig` dataclass and update configuration files.
- [ ] Refactor `trainer.initial_state` / `train_lm` to use a unified initialization function based on `InitConfig`.
- [ ] Update checkpoint saving/loading utilities to persist model (or full training) config.
- [ ] Provide shims translating old flags to the new configuration and emit deprecation warnings.
- [ ] Add tests covering:
    - fresh training run
    - resuming from latest checkpoint
    - initializing from another run with/without optimizer state
    - initializing from HF checkpoint
    - initialization with data reset
- [ ] Update documentation and examples to reference the new initialization flow.
