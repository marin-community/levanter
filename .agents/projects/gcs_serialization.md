# GCS TensorStore Loader Revamp

## Background

The current GCS safetensors loader asks TensorStore for one slice per FSDP shard. Each slice triggers a tiny random read across the safetensors file, so multi-host checkpoints make millions of independent byte-range requests. Even with batching, the async path ends up waiting on one slice at a time, yielding very slow staging relative to copying the full files locally.

We want to keep the “no local copy” property for TPU jobs (local disk is scarce) while avoiding the pathological access pattern. Safetensors provides the metadata we need: each tensor lives in a flat file with known dtype, shape, and byte offsets. We can leverage that to schedule a small number of large, contiguous reads and shuffle the bytes to the devices we actually need.

## Constraints & Targets

- Do not require staging the entire checkpoint to local disk; operate directly on GCS via `fsspec`.
- Preserve existing HF checkpoint interfaces (`create_async_array_from_callback` consumers should keep working).
- Support multi-host setups using the existing device mesh; downstream code assumes tensors arrive with shardings from `best_effort_sharding`.
- Read complete tensors (never split a single key across multiple chunks) to keep bookkeeping manageable.
- Default chunk size: 2 GB, but expose an override (env or config) so we can tune for memory pressure.
- Keep the legacy small-slice loader behind a feature flag for rollback while the new path hardens.

## Proposed Algorithm

1. **Metadata pass (all hosts).**
   - Use the safetensors header to build `TensorMeta(key, file_path, dtype, byte_start, byte_end, shape)`.
   - Cache file-level metadata so repeated loads do not re-download headers.

2. **Chunk construction (all hosts, deterministic).**
   - For each file, walk tensors in storage order and pack them into contiguous `ChunkSpec`s.
   - Each `ChunkSpec` keeps: `file_path`, `byte_start`, `byte_end`, ordered list of `(key, dtype, shape, offset_within_chunk)`, and the `PartitionSpec` for each key (via `best_effort_sharding`).
   - Never split a tensor across chunks; if one tensor exceeds the chunk size limit, let the chunk size grow to accommodate it.

3. **Chunk assignment (all hosts, deterministic).**
   - Enumerate chunks globally (e.g., sorted by `(file_path, byte_start)`).
   - Assign each chunk to a host `owner_rank = chunk_index % world_size` (or similar simple rule) so every process arrives at the same mapping without communication.

4. **Chunk materialization (owner host only).**
   - Owner performs a single `fsspec` range read for `[byte_start, byte_end)` and converts the result to a NumPy buffer.
   - Produce views for each tensor using the metadata offsets; apply dtype conversion if requested.

5. **Hand-off via callbacks (future refinement).**
   - Initial goal was to wrap each chunk in `jax.make_array_from_callback`, returning real data on the owning host and lightweight placeholders elsewhere.
   - Current implementation shortcuts this by broadcasting the NumPy buffer to every host via `multihost_utils.broadcast_one_to_all`, so each process materialises the full chunk locally. This keeps the code simple but duplicates host RAM usage; revisit once we need leaner per-host footprints.

6. **Per-key extraction.**
   - After reshaping, cut the chunk back into individual tensors using the recorded offsets.
   - Supply each tensor to the existing `create_async_array_from_callback` path, preserving the computed sharding.

7. **Cleanup / Feature flag.**
   - Provide a configuration flag (env var `LEVANTER_USE_CHUNKED_GCS_LOADER`, default off initially) to pick between loaders.
   - Ensure we can fall back if issues arise during rollout.

### Pseudocode Sketch

```python
def build_chunk_specs(tensor_meta, chunk_limit):
    specs = []
    current = new_chunk()
    for meta in sorted(tensor_meta, key=lambda m: (m.file_path, m.byte_start)):
        if meta.file_path != current.file_path or current.size + meta.size > chunk_limit:
            specs.append(current)
            current = new_chunk(meta.file_path)
        current.add_tensor(meta)
    if current.tensors:
        specs.append(current)
    return specs

def chunk_owner(chunk_index, world_size):
    return chunk_index % world_size
```

## Current Status

- Implemented `SafetensorChunkLoader` that parses metadata, builds deterministic chunk specs, and materialises/broadcasts chunks across hosts before slicing them into per-key NumPy arrays.
- HF checkpoint loading now streams through the chunk loader and places tensors on devices using `best_effort_sharding`.
- Unit tests cover basic round-trip, dtype override handling, and single-tensor fetch on a local filesystem.
- Simplification: every host receives the full chunk via `broadcast_one_to_all`; optimisation to avoid duplicate host allocations is deferred.

## Implementation Tasks

- [x] Parse safetensors metadata on every worker and build `{key: TensorMeta}` without redundant I/O.
- [x] Add chunk builder producing deterministic `ChunkSpec`s that never split tensors and respect the configured size limit (default 2 GB).
- [x] Implement deterministic host assignment (`chunk_index % world_size`) and expose helper to query chunk ownership.
- [x] Write owner-side reader that performs a single ranged `fsspec` read, converts to NumPy, and slices out per-key buffers.
- [ ] Reintroduce a callback-based hand-off (or equivalent) that keeps non-owners from allocating each chunk, while still integrating with `best_effort_sharding`.
- [x] Reconstruct individual tensors from the chunk buffer, apply dtype conversions, and pass them to existing consumers with their target shardings.
- [ ] Add per-process chunk prefetch so the next chunk begins materialising while the current tensors are extracted.
- [ ] Add configuration toggles (feature flag, chunk size, read concurrency) and document defaults.
- [ ] Keep/restore the legacy fine-grained loader behind a feature flag for rollback.
- [ ] Add broader tests (multi-host mock or integration) to verify redistribution behaviour, dtype handling, and the fallback switch.

## Open Questions / Follow-Ups

- Validate memory pressure on TPU hosts for 2GB reads; adjust defaults if necessary.
- Confirm any future placeholder/callback approach doesn’t confuse downstream shape inference; document expected array shapes.
- Decide where to log chunk assignments / timings for profiling.
- Consider caching chunk specs / metadata between runs to avoid recomputation when repeatedly loading the same checkpoint revision.
- Investigate alternatives to the current broadcast to keep per-host memory proportional to local shard needs.
