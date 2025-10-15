# Levanter Training Pipeline Animation Storyboard

This storyboard lists the sequence of panels that the training animation will
highlight. Each panel pairs a high-level step with the concrete actions that
Levanter performs during language-model training.

1. **Load Configuration**
   - Parse base YAML/JSON configs with `draccus.load`.
   - Apply CLI or sweep overrides and materialize experiment IDs.
   - Seed PRNG streams and initialize logging/tracker metadata.
2. **Assemble Mesh Resources**
   - Probe TPU/GPU topology and decide mesh axis sizes.
   - Instantiate Haliax `Axis` objects for batch/model/heads dimensions.
   - Create PJIT device mesh and register it with Haliax axis registry.
3. **Prepare Dataset Stream**
   - Expand dataset config into concrete shards (GCS/S3/local).
   - Tokenize examples with the configured tokenizer and pack sequences.
   - Build `AsyncDataset` pipeline with map/shuffle/prefetch transforms.
   - Stage batches into device-host queues for overlap with compute.
4. **Build Model & Optimizer**
   - Instantiate tokenizer-dependent embeddings and Transformer blocks
     using Equinox modules parameterized by named axes.
   - Initialize parameter PyTrees with PRNG keys split per layer.
   - Configure optimizer (Adafactor/AdamW) and learning-rate schedule.
5. **Distribute State**
   - Shard parameters and optimizer slots across the mesh with `hax.shard`.
   - Broadcast state dicts and master weights with `jax.device_put`.
   - Scatter per-device PRNGs and gradient-accumulation buffers.
6. **Fetch Next Batch**
   - Pull pre-tokenized sequences from the async iterator.
   - Pad or pack sequences to match mesh-aligned batch shapes.
   - Transfer batches to device memory using double buffering.
7. **Forward Pass**
   - Embed tokens, add positional encodings, and apply dropout if enabled.
   - Run stacked Transformer layers under `jax.jit` or `eqx.filter_jit`.
   - Project hidden states through LM head and compute logits.
8. **Compute Loss**
   - Align logits/targets, mask padding tokens, and compute cross-entropy.
   - Aggregate auxiliary stats (perplexity, token accuracy) per axis.
   - Reduce metrics across devices via `hax.mean` / `jax.lax.pmean`.
9. **Backward Pass**
   - Use `jax.value_and_grad` or `eqx.filter_value_and_grad` to get loss
     and gradients.
   - Apply gradient clipping, weight decay prep, and optional grad scaling.
   - Collect gradient norms and microbatch stats for logging.
10. **Optimizer Update**
    - Update optimizer slots (momentum, variance, Adafactor statistics).
    - Apply parameter deltas respecting partition specs and weight decay.
    - Advance schedulers (warmup, cosine decay) and reset grad buffers.
11. **Checkpoint & Log**
    - Stream metrics to tracker hooks (WandB, TensorBoard, JSONL).
    - Trigger evaluation windows and validation datasets if scheduled.
    - Write async checkpoints with parameters, optimizer state, and metadata.
12. **Loop Control**
    - Increment global step counters and wall-clock accounting.
    - Check stop conditions: max steps, tokens, time, or convergence.
    - Schedule next iteration or initiate graceful shutdown/cleanup.

The animation will reveal each panel sequentially, keeping prior steps dimmed
for context so viewers can follow the end-to-end flow.

## Rendering the Animation

1. Ensure `matplotlib` is available in your environment. With `uv`, run
   ``UV_CACHE_DIR=.uv-cache uv pip install matplotlib`` from the repository
   root to keep caches inside the workspace.
2. Generate the animation with
   ``UV_CACHE_DIR=.uv-cache uv run python docs/figures/generate_training_pipeline_animation.py``.
   The default output is `docs/figures/training_pipeline_animation.gif`.
3. Use the `--output` flag to write to a different path or switch to MP4 by
   supplying a `.mp4` suffix (requires `ffmpeg`).
4. Adjust pacing with `--hold-frames` (higher values linger on each panel) and
   `--fps` to control animation speed.
