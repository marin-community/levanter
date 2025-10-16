# Levanter Inference System Architecture

**Author**: Claude (AI Assistant)
**Date**: 2025-10-16
**Perspective**: Tracing a batch through the inference pipeline from entry to completion

---

## Table of Contents

1. [Overview](#overview)
2. [Core Data Structures](#core-data-structures)
3. [The Batch Journey](#the-batch-journey)
4. [Detailed Component Walkthrough](#detailed-component-walkthrough)
5. [Memory Management & Page System](#memory-management--page-system)
6. [Performance Considerations](#performance-considerations)

---

## Overview

The Levanter inference system is a high-performance, batched inference engine for language models. It uses **paged attention** (similar to vLLM/PagedAttention) to efficiently manage KV cache memory and supports:

- **Continuous batching**: Dynamic admission of new requests while others are being processed
- **Multi-sample generation**: Efficient `n>1` generation via sequence cloning with copy-on-write semantics
- **Prefill and decode separation**: Distinct handling of prompt processing vs. autoregressive generation
- **JIT compilation**: All critical paths are JIT-compiled for maximum throughput

The system is organized into five key modules:
- **`engine.py`**: The orchestrator (`InferenceEngine` class)
- **`jit_scheduler.py`**: On-device state management (`DecodeState`, `TokenQueue`, `SequenceTable`)
- **`page_table.py`**: Memory allocation for KV cache (`PageTable`, `PageBatchInfo`)
- **`utils.py`**: Helper functions for validity checking, array manipulation
- **`openai.py`**: HTTP API wrapper for OpenAI-compatible serving

---

## Core Data Structures

### 1. `InferenceEngine` (engine.py:685)

The main user-facing class. Manages:
- **Model & tokenizer**: The LM and its tokenizer
- **GenState**: Device-side generation state (cache + decode state)
- **Host-side tracking**: Free slot management, request queue, result accumulation
- **Configuration**: Memory limits, batch sizes, decode parameters

Key methods:
- `generate(requests)`: Main entry point for batch generation
- `_admit_from_queue()`: Tries to admit requests from queue if capacity allows
- `_prefill_prompts(requests)`: Packs prompts into PrefillWork structure
- `_ingest_outputs(outputs)`: Extracts completed tokens from device and updates host state

### 2. `GenState` (engine.py:180)

Container for device-side generation state:
- **`cache: PageCache`**: The KV cache (model-specific, paged structure)
- **`decode_state: DecodeState`**: All sequence metadata and token queues

Provides `clone_sequence()` for efficient multi-sample generation (copies page references, creates fresh page for partial tail).

### 3. `DecodeState` (jit_scheduler.py:544)

The heart of on-device state management. Tracks:
- **`tokens`**: Buffer of generated tokens per sequence `[seq, position]`
- **`logprobs`**: Optional log probabilities per token
- **`sequences: SequenceTable`**: Aggregated per-sequence metadata
- **`page_table: PageTable`**: Global page allocator
- **`tqueue: TokenQueue`**: Pending tokens waiting to be decoded
- **`finished`**: Boolean mask for completed sequences
- **Per-sequence parameters**: `max_num_tokens`, `stop_tokens`, `temperature`, `prng_keys`

Key operations:
- `reserve_slot(slot_id)`: Allocate a sequence slot
- `assign_seq(...)`: Initialize a new sequence with prompt tokens
- `allocate_for_seq(token_slot_ids, token_pos_ids)`: Allocate KV pages for tokens
- `pack_next_sequence(max_tokens)`: Dequeue tokens from queue and pack them for decode
- `update_tokens(new_tokens, slot_ids, log_probs, num)`: Append newly sampled tokens

### 4. `SequenceTable` (jit_scheduler.py:74)

Compact metadata for active sequences:
- **`seq_lens`**: Current length of each sequence
- **`clone_sources`**: For clones, the parent sequence ID (INVALID if primary)
- **`kv_pages`**: Physical page assignments per sequence
- **`page_indices`**: Same as kv_pages (used during allocation)
- **`used_mask`**: Which slots are active

Handles:
- Slot reservation and release
- Page allocation coordination with `PageTable`
- Page freeing when sequences finish
- Clone page copying (increments ref counts, allocates fresh tail page)

### 5. `PageTable` (page_table.py:25)

Global KV page allocator:
- **`page_ref_counts`**: Reference count per page (0 = free, >0 = shared/used)
- **`page_size`**: Tokens per page (typically 128)
- **`max_seqs`**: Maximum concurrent sequences
- **`pages_per_seq`**: Maximum pages per sequence

Uses reference counting to support copy-on-write page sharing for clones.

### 6. `TokenQueue` (jit_scheduler.py:973)

FIFO queue for tokens waiting to be processed:
- **`queued_tokens`**: Flat array of pending token IDs
- **`queued_slot_ids`**: Which sequence each token belongs to
- **`queued_pos_ids`**: Absolute position of each token in its sequence
- **`num_queued_tokens`**: Current queue occupancy

Operations:
- `enqueue_tokens(...)`: Append new tokens to queue
- `pack_next_sequence(max_tokens)`: Dequeue and sort tokens by slot ID for batching
- `purge_queue_of_slot(slot_id)`: Remove all tokens for a finished sequence

### 7. `PrefillWork` (engine.py:277)

Host-side description of prefill work to be executed on device:
- **`queue: TokenQueue`**: Prompt tokens packed for prefill
- **`new_slot_ids`**: Which slots are being initialized
- **`clone_targets`**: For clones, the parent slot ID (INVALID for primaries)
- **`prompt_tokens`**: Actual prompt token arrays `[seq, position]`
- **`prompt_lengths`**: Length of each prompt
- **`seq_params: SeqDecodingParams`**: Per-sequence generation parameters

Built on the host, then passed to `_run_prefill()` which executes on device.

### 8. `PageBatchInfo` (page_table.py:58)

Describes a batch of sequences for model.decode():
- **`slot_ids`**: Which sequence slots are in this batch
- **`page_indices`**: KV page assignments per sequence
- **`seq_lens`**: Current length of each sequence
- **`cu_q_lens`**: Cumulative query lengths (for packed attention)
- **`num_seqs`**: Number of sequences in batch
- **`new_token_dests`**: Flattened KV cache destinations for new tokens

Used by the model to know where to read/write KV cache entries.

---

## The Batch Journey

Let's trace a batch of requests from entry to completion.

### Phase 0: Initialization

```python
# User creates an engine
engine = InferenceEngine.from_model_with_config(model, tokenizer, config)
```

At this point:
- `PageTable` is initialized with `max_pages` pages (all free)
- `DecodeState` is initialized with empty sequence slots
- `GenState` wraps the KV cache and decode state
- `InferenceEngine` maintains a host-side free slot list

### Phase 1: Request Submission

```python
# User submits requests
requests = [
    Request(
        prompt_tokens=[1, 2, 3, 4],
        request_id=0,
        decode_params=SeqDecodingParams(...),
        n_generations=2,  # Want 2 samples
        enable_logprobs=True
    ),
    Request(prompt_tokens=[5, 6, 7], request_id=1, decode_params=..., n_generations=1)
]

result = engine.generate(requests)
```

**Entry point**: `InferenceEngine.generate()` (engine.py:1056)

1. **Validation**: Check that no request needs more than `max_seqs` slots
2. **Enqueue**: `self.enqueue_requests(requests)` adds to `self.request_queue`
3. **Initialize result tracking**: Create empty `DecodeResult` buckets per (request, choice)

### Phase 2: Admission & Prefill Preparation

**Function**: `InferenceEngine._admit_from_queue()` (engine.py:844)

The engine simulates resource usage to determine which requests can fit:

```python
sim_slots = len(self.free_slots)  # e.g., 256 available
sim_pages = self._free_page_count()  # e.g., 1000 pages free
sim_tokens = 0
max_prefill_size = 4096  # Max tokens in one prefill batch

batch = []
while self.request_queue:
    nxt = self.request_queue[0]
    need_slots = nxt.n_generations  # e.g., 2 for request 0
    need_pages = (len(prompt) + page_size - 1) // page_size

    if sim_slots >= need_slots and sim_pages >= need_pages and sim_tokens + len(prompt) <= max_prefill_size:
        batch.append(self.request_queue.popleft())
        sim_slots -= need_slots
        sim_pages -= need_pages
        sim_tokens += len(prompt)
    else:
        break  # Can't fit more
```

**Outcome**: `batch = [Request(id=0, n=2), Request(id=1, n=1)]` (both fit)

### Phase 3: Packing Prefill Work

**Function**: `InferenceEngine._prefill_prompts(batch)` (engine.py:902)

This builds `PrefillWork` on the **host** (CPU), preparing data for device execution:

```python
# Allocate buffers
queue_tokens = np.full((max_prefill_size,), INVALID, dtype=np.int32)
queue_slot_ids = np.full((max_prefill_size,), INVALID, dtype=np.int32)
queue_pos_ids = np.full((max_prefill_size,), INVALID, dtype=np.int32)

work_slot_ids = np.full((max_slots,), INVALID, dtype=np.int32)
clone_targets = np.full((max_slots,), INVALID, dtype=np.int32)
prompt_tokens = np.full((max_slots, max_seq_len), INVALID, dtype=np.int32)
prompt_lengths = np.zeros((max_slots,), dtype=np.int32)

# For each request
offset = 0
total_new = 0

# Request 0: [1,2,3,4], n=2
slot_primary = self.free_slots.pop()  # e.g., 0
queue_tokens[0:4] = [1,2,3,4]
queue_slot_ids[0:4] = 0
queue_pos_ids[0:4] = [0,1,2,3]

work_slot_ids[0] = 0
clone_targets[0] = INVALID  # Primary
prompt_tokens[0, 0:4] = [1,2,3,4]
prompt_lengths[0] = 4
# ...set seq_params for slot 0

self.local_map[0] = (0, 0)  # request_id=0, choice=0
self.sequences[0] = {0: 0}

offset += 4
total_new += 1

# Clone for n=2
slot_clone = self.free_slots.pop()  # e.g., 1
work_slot_ids[1] = 1
clone_targets[1] = 0  # Clone of slot 0
prompt_lengths[1] = 4
# ...set seq_params with different PRNG key

self.local_map[1] = (0, 1)  # request_id=0, choice=1
self.sequences[0][1] = 1

total_new += 1

# Request 1: [5,6,7], n=1
slot_primary = self.free_slots.pop()  # e.g., 2
queue_tokens[4:7] = [5,6,7]
queue_slot_ids[4:7] = 2
queue_pos_ids[4:7] = [0,1,2]

work_slot_ids[2] = 2
clone_targets[2] = INVALID
prompt_tokens[2, 0:3] = [5,6,7]
prompt_lengths[2] = 3
# ...set seq_params

self.local_map[2] = (1, 0)
self.sequences[1] = {0: 2}

offset += 3
total_new += 1
```

**Result**: `PrefillWork` with:
- `queue`: 7 tokens packed: `[1,2,3,4,5,6,7]` with slot IDs `[0,0,0,0,2,2,2]`
- `new_slot_ids`: `[0, 1, 2, ...]`
- `clone_targets`: `[INVALID, 0, INVALID, ...]` (slot 1 is a clone of slot 0)
- `prompt_tokens`, `prompt_lengths`, `seq_params`: All packed

### Phase 4: Device Prefill Execution

**Function**: `_run_prefill()` (engine.py:468, JIT-compiled)

This runs on the **device** (GPU/TPU). It has two stages:

#### Stage 4a: Apply Prefill Work

**Function**: `_apply_prefill_work(gen_state, work)` (engine.py:425)

Uses a `fori_loop` to process each slot instruction:

```python
for i in range(total_new):  # 0, 1, 2
    slot_val = work.new_slot_ids[i]
    parent_val = work.clone_targets[i]

    if is_valid(parent_val):  # i=1 (clone)
        # Clone slot 0 -> slot 1
        gen_state, _ = gen_state.clone_sequence(parent_val, slot_val, seq_params)
        # This:
        # - Reserves slot 1 in DecodeState
        # - Copies tokens from slot 0
        # - Increments ref counts on shared pages
        # - Allocates fresh page for partial tail
        # - Sets clone_source[1] = 0
    else:  # i=0, i=2 (primaries)
        # Reserve slot and assign prompt
        decode_state, assigned = decode_state.reserve_slot(slot_val)
        decode_state = decode_state.assign_seq(
            local_slot_id=slot_val,
            tokens=work.prompt_tokens[i],
            seq_len=work.prompt_lengths[i],
            seq_params=seq_params
        )
        # This sets tokens[slot_val, 0:prompt_len] and seq_lens[slot_val] = prompt_len
```

After this stage:
- Slots 0, 1, 2 are reserved and initialized
- Slot 1 is marked as a clone of slot 0
- No pages allocated yet (happens in next stage)

#### Stage 4b: Prefill Kernel

**Function**: `_prefill_kernel(gen_state, model, sampler, queue, max_seqs_in_prefill)` (engine.py:321)

This is where the model actually runs:

```python
tokens = queue.queued_tokens  # [1,2,3,4,5,6,7]
pos_ids = queue.queued_pos_ids  # [0,1,2,3,0,1,2]
slot_ids = queue.queued_slot_ids  # [0,0,0,0,2,2,2]

# 1. Allocate KV pages
decode_state, binfo = decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)
```

**Deep dive**: `DecodeState.allocate_for_seq()` calls `SequenceTable.allocate_for_seq()` (jit_scheduler.py:211):

1. **Group tokens by sequence**: Uses `jax.ops.segment_sum/max` to compute per-sequence token counts and max positions
   - Sequence 0: 4 tokens, max_pos=3
   - Sequence 2: 3 tokens, max_pos=2
   - (Sequence 1 is a clone, already has pages from slot 0)

2. **Update seq_lens**: `new_lens[seq_id] = max(current_len, max_pos + 1)`
   - seq_lens[0] = 4, seq_lens[2] = 3

3. **Calculate page needs**:
   - Slot 0: needs `(4 + 127) // 128 = 1` page, currently has 0 → allocate 1
   - Slot 2: needs `(3 + 127) // 128 = 1` page, currently has 0 → allocate 1

4. **Allocate pages**: For each sequence needing pages:
   ```python
   for seq in [0, 2]:
       for page_idx in range(old_needed, new_needed):
           free_page = argmin(page_ref_counts)  # Find free page
           page_ref_counts[free_page] += 1
           page_indices[seq, page_idx] = free_page
   ```
   - Slot 0 gets page 0 (ref_count[0] = 1)
   - Slot 2 gets page 1 (ref_count[1] = 1)
   - (Slot 1 already shares page 0 from clone, ref_count[0] = 2 now)

5. **Build PageBatchInfo**: Compute token destinations in flattened KV cache:
   ```python
   token_dests = []
   for i, (seq_id, pos_id) in enumerate(zip(slot_ids, pos_ids)):
       page_idx = pos_id // page_size
       page_offset = pos_id % page_size
       page = page_indices[seq_id, page_idx]
       dest = page * page_size + page_offset
       token_dests.append(dest)
   ```
   Result: `token_dests = [0, 1, 2, 3, 128, 129, 130]` (page 0 offsets 0-3, page 1 offsets 0-2)

**Back to prefill_kernel**:

```python
# 2. Compute sample boundaries
sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_seqs_in_prefill)
# Boundaries are where pos_id == seq_len - 1
# For slot 0: boundary at position 3 (pos_ids[3] == seq_lens[0] - 1)
# For slot 2: boundary at position 6 (pos_ids[6] == seq_lens[2] - 1)
# sample_indices = [3, 6, INVALID, ...]

# 3. Run model
logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
# tokens: [1,2,3,4,5,6,7], length 7
# logits: [vocab_size] for each position → [7, vocab_size]
# cache: Updated KV cache with new values written to token_dests

# 4. Sample at boundaries
logits_at_samples = logits[sample_indices]  # [2, vocab_size] (positions 3 and 6)
num_new_tokens = 2

new_slot_ids = slot_ids[sample_indices]  # [0, 2]
new_pos_ids = pos_ids[sample_indices]  # [3, 2]
prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

temps = decode_state.temperature[new_slot_ids]
new_tokens, log_probs = hax.vmap(sampler)(logits_at_samples, temps, key=prng_keys)
# new_tokens: [token_0, token_2] (sampled for slots 0 and 2)

# 5. Update state and enqueue
decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)
```

**Deep dive**: `DecodeState.update_tokens()` (jit_scheduler.py:864):

```python
for i in range(num_new_tokens):  # 0, 1
    sid = local_slot_ids[i]  # 0, then 2
    pos = seq_lens[sid]  # 4, then 3

    # Append token
    tokens[sid, pos] = new_tokens[i]
    if logprobs is not None:
        logprobs[sid, pos] = new_log_probs[i]

    # Increment length
    seq_lens[sid] += 1  # Now 5, then 4

    # Check for completion
    max_allowed = max_num_tokens[sid]
    len_done = (pos + 1) >= max_allowed
    stop_done = check_stop_tokens(...)
    finished[sid] = len_done | stop_done

    # Record position for queue
    pos_ids[i] = pos

    # If finished, mark for purging
    should_purge[i] = finished[sid]

# Purge finished sequences from new_tokens (if any)
new_tokens = purge(new_tokens, should_purge)
new_slot_ids = purge(new_slot_ids, should_purge)
pos_ids = purge(pos_ids, should_purge)

# Enqueue remaining tokens
tqueue = tqueue.enqueue_tokens(new_tokens, new_slot_ids, pos_ids, num_remaining)
```

Assuming none finished yet:
- `tokens[0, 4] = token_0`, `seq_lens[0] = 5`
- `tokens[2, 3] = token_2`, `seq_lens[2] = 4`
- `tqueue.queued_tokens[0:2] = [token_0, token_2]`
- `tqueue.queued_slot_ids[0:2] = [0, 2]`
- `tqueue.queued_pos_ids[0:2] = [4, 3]`
- `tqueue.num_queued_tokens = 2`

```python
# 6. Create outputs buffer and append
outputs = _DecodeOutputs.init(max_tokens=..., max_seqs=...)
outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)
# outputs.tokens[0:2] = [token_0, token_2]
# outputs.slot_ids[0:2] = [0, 2]
# outputs.logprobs[0:2] = [lp_0, lp_2]
# outputs.num_tokens = 2

# 7. Handle clones (if any)
if decode_state.clone_sources is not None:
    gen_state, outputs = _handle_clones(gen_state, logits_at_samples, new_slot_ids, new_pos_ids, sampler, outputs)
```

**Deep dive**: `_handle_clones()` (engine.py:479):

```python
# Find which clones have their parent in this batch
# clone_sources = [INVALID, 0, INVALID, ...] (slot 1 is clone of slot 0)
# new_slot_ids = [0, 2]

for i, src in enumerate(clone_sources):  # i=1, src=0
    if is_valid(src):
        # Find src in new_slot_ids
        idx = find_index(new_slot_ids, src)  # idx=0 (slot 0 is at position 0)
        if is_valid(idx):
            source_indices[i] = idx

# source_indices = [INVALID, 0, INVALID, ...]
can_sample = source_indices != INVALID  # [False, True, False, ...]

# Compact list of clones to process
selected = where(can_sample)  # [1]
num_new = 1

# Gather data for clones
tgt_ids = [1]
src_pos = source_indices[selected] = [0]
src_ids = new_slot_ids[src_pos] = [0]
logits_this_time = logits_at_samples[src_pos]  # Reuse parent's logits!
pos_ids_this_time = new_pos_ids[src_pos] = [4]

# Sample with different key
temps = decode_state.temperature[tgt_ids]  # temp for slot 1
prng_keys = decode_state.prng_keys_for(tgt_ids, pos_ids_this_time)  # Different key!

clone_tokens, clone_log_probs = hax.vmap(sampler)(logits_this_time, temps, key=prng_keys)
# clone_tokens = [token_1_clone] (different sample from same logits)

# Copy pages from parent to clone (ref count management)
for i in range(num_new):  # i=0
    src_slot = src_ids[i] = 0
    dst_slot = tgt_ids[i] = 1
    decode_state = decode_state.clone_pages_from(src_slot, dst_slot)
    # This updates page_indices[1] to match page_indices[0]
    # Increments ref counts on shared pages
    # If there's a partial tail page, copies it to a fresh page

# Update tokens for clones
decode_state = decode_state.update_tokens(clone_tokens, tgt_ids, clone_log_probs, num_new)
# tokens[1, 4] = token_1_clone, seq_lens[1] = 5
# Enqueues to tqueue: queued_tokens[2] = token_1_clone, queued_slot_ids[2] = 1

# Discharge clones so they're not reprocessed
decode_state = decode_state.discharge_clone(tgt_ids, num_new)
# clone_sources[1] = INVALID (no longer pending)

# Append to outputs
outputs = outputs.append(clone_tokens, tgt_ids, clone_log_probs, num_new, decode_state.finished)
# outputs.tokens[2] = token_1_clone
# outputs.slot_ids[2] = 1
# outputs.num_tokens = 3
```

**Result of prefill**:
- `outputs.tokens = [token_0, token_2, token_1_clone]`
- `outputs.slot_ids = [0, 2, 1]`
- `outputs.num_tokens = 3`
- `tqueue.queued_tokens = [token_0, token_2, token_1_clone]`
- `tqueue.queued_slot_ids = [0, 2, 1]`
- `tqueue.queued_pos_ids = [4, 3, 4]`
- `tqueue.num_queued_tokens = 3`
- All sequences active, none finished yet

### Phase 5: Host-Side Output Extraction

**Function**: `InferenceEngine._ingest_outputs(outputs)` (engine.py:1358)

Back on the host, we extract tokens from the prefill outputs:

```python
outputs = jax.device_get(outputs)  # Transfer from device to host
n = outputs.num_tokens  # 3

for i in range(n):  # 0, 1, 2
    local_slot = outputs.slot_ids[i]  # 0, 2, 1
    tok = outputs.tokens[i]

    info = self.local_map[local_slot]
    # local_map[0] = (0, 0) → request_id=0, choice=0
    # local_map[2] = (1, 0) → request_id=1, choice=0
    # local_map[1] = (0, 1) → request_id=0, choice=1

    rid, cid = info
    dr = self.results[rid][cid]
    dr.token_list.append(tok)
    dr.logprobs.append(outputs.logprobs[i])
    dr.tokens_decoded += 1

# Update done flags
for local_slot, is_done in enumerate(outputs.finished):
    if is_done:
        self.results[rid][cid].done = True
```

**Result**:
- `results[0][0].token_list = [token_0]`
- `results[0][1].token_list = [token_1_clone]`
- `results[1][0].token_list = [token_2]`

### Phase 6: Autoregressive Generation Loop

**Function**: `_run_generation_loop(gen_state, model, sampler, max_tokens_per_round, max_rounds)` (engine.py:594, JIT-compiled)

This is a `while_loop` that continues until all sequences finish or max_rounds reached:

```python
def cond(state):
    gen_state, outputs, step = state
    return (
        (step < max_rounds) &
        (gen_state.decode_state.num_queued_tokens > 0) &
        (~hax.all(gen_state.decode_state.finished))
    )

def body(state):
    gen_state, outputs, step = state

    # 1. Pack next batch
    decode_state, packed_seq = gen_state.decode_state.pack_next_sequence(max_tokens_per_round)
```

**Deep dive**: `DecodeState.pack_next_sequence()` forwards to `TokenQueue.pack_next_sequence()` (jit_scheduler.py:1049):

```python
max_tokens = 256  # Example
num = min(num_queued_tokens, max_tokens)  # min(3, 256) = 3

# Dequeue first 3 tokens
tokens = queued_tokens[0:3]  # [token_0, token_2, token_1_clone]
slot_ids = queued_slot_ids[0:3]  # [0, 2, 1]
pos_ids = queued_pos_ids[0:3]  # [4, 3, 4]

# Roll queue to remove dequeued items
rolled_tokens = roll(queued_tokens, -3)
# [..., INVALID, INVALID, INVALID]
rolled_slot_ids = roll(queued_slot_ids, -3)
rolled_pos_ids = roll(queued_pos_ids, -3)

# Update queue
num_queued_tokens -= 3  # Now 0

# Sort by slot_id for efficient batching
sort_order = argsort(slot_ids)  # [0, 2, 1] → [0, 1, 2] (slot order)
tokens = tokens[sort_order]  # [token_0, token_1_clone, token_2]
slot_ids = slot_ids[sort_order]  # [0, 1, 2]
pos_ids = pos_ids[sort_order]  # [4, 4, 3]

return PackedSequence(tokens=tokens, slot_ids=slot_ids, pos_ids=pos_ids, num_tokens=3)
```

**Back to body**:

```python
    tokens = packed_seq.tokens  # [token_0, token_1_clone, token_2]
    pos_ids = packed_seq.pos_ids  # [4, 4, 3]
    slot_ids = packed_seq.slot_ids  # [0, 1, 2]

    # 2. Allocate pages (if needed)
    decode_state, binfo = decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)
    # All sequences already have sufficient pages for these positions, so no new allocations
    # binfo.new_token_dests = [4, 132, 259] (page 0 offset 4, page 1 offset 4, page 2 offset 3)

    # 3. Compute sample boundaries
    sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_tokens_per_round)
    # All tokens are at their sequence boundary (pos_id == seq_len - 1)
    # sample_indices = [0, 1, 2]

    # 4. Decode
    logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
    # logits: [3, vocab_size]
    logits_at_samples = logits[sample_indices]  # [3, vocab_size]

    # 5. Sample
    num_new_tokens = 3
    new_slot_ids = slot_ids[sample_indices]  # [0, 1, 2]
    new_pos_ids = pos_ids[sample_indices]  # [4, 4, 3]
    prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)
    temps = decode_state.temperature[new_slot_ids]

    new_tokens, log_probs = hax.vmap(sampler)(logits_at_samples, temps, key=prng_keys)
    # new_tokens = [next_token_0, next_token_1, next_token_2]

    # 6. Update state
    decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)
    # Appends tokens, increments seq_lens, checks completion, enqueues to tqueue
    # If any sequence finishes, it's marked in finished[] and purged from enqueue

    new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)

    # 7. Append to outputs
    outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)

    # 8. Release finished sequences (device-side)
    new_gen_state = _release_finished_device(new_gen_state)
    # Frees pages and invalidates finished slots

    return new_gen_state, outputs, step + 1
```

**Loop continues**: This repeats for each decode iteration:
- Pack next batch from queue
- Allocate pages if needed
- Run model.decode()
- Sample new tokens
- Update state and enqueue
- Check for completion

Each iteration produces new tokens, which are accumulated in `outputs` and extracted on the host.

### Phase 7: Final Output Extraction & Result Assembly

**Function**: `InferenceEngine.generate()` (engine.py:1056, continued)

After the generation loop finishes:

```python
# In the main generate() method
while not _all_done():
    # Run generation loop
    future_state, decode_outputs = _run_generation_loop(gen_state, model, sampler, max_tokens_per_round, max_rounds)
    self.gen_state = future_state

    # Extract outputs
    new_tokens = self._ingest_outputs(decode_outputs)
    # This updates self.results with new tokens

    # Release finished sequences (host-side)
    self._release_finished_sequences(decode_outputs)
    # This frees slots from self.local_map and self.sequences
    # Adds freed slots back to self.free_slots

    # Try to admit more requests
    admit_outputs = self._admit_from_queue()
    if admit_outputs:
        self._ingest_outputs(admit_outputs)

    # Check if all expected sequences are done
    if _all_done():
        break

# Assemble final results
outputs_list = []
logprobs_list = []
for r in requests:
    rid = r.request_id
    for k in range(r.n_generations):
        dr = self.results[rid][k]
        outputs_list.append(dr.token_list)
        logprobs_list.append(dr.logprobs)

return GenerationResult(tokens=outputs_list, logprobs=logprobs_list, total_generated=...)
```

**Final result**:
```python
GenerationResult(
    tokens=[
        [token_0, next_token_0, ...],  # Request 0, choice 0
        [token_1_clone, next_token_1, ...],  # Request 0, choice 1
        [token_2, next_token_2, ...]  # Request 1, choice 0
    ],
    logprobs=[
        [lp_0, lp_next_0, ...],
        [lp_1_clone, lp_next_1, ...],
        [lp_2, lp_next_2, ...]
    ],
    total_generated=total_token_count
)
```

---

## Detailed Component Walkthrough

### Sequence Lifecycle

1. **Reservation**: `DecodeState.reserve_slot()` → Allocates a slot, marks `used_mask[slot]=True`
2. **Initialization**: `DecodeState.assign_seq()` → Sets prompt tokens, seq_len, parameters
3. **Page Allocation**: `allocate_for_seq()` → Allocates KV pages as tokens are processed
4. **Token Generation**: Loop of `pack_next_sequence()` → `model.decode()` → `update_tokens()`
5. **Completion Detection**: When `seq_len >= max_num_tokens` or stop token matched
6. **Marking Finished**: `finished[slot] = True`, tokens purged from queue
7. **Device-Side Release**: `_release_finished_device()` → Frees pages, invalidates slot
8. **Host-Side Release**: `_release_finished_sequences()` → Updates local_map, returns slot to free_slots

### Clone Mechanics

**Why clones?** For `n>1` generation, we want to share the prompt processing and KV cache for efficiency.

**How it works**:
1. **Primary sequence**: Processed normally through prefill
2. **Clone creation**: After primary samples first token, clone is created:
   - `clone_sequence()` called with parent and child slot IDs
   - Copies token buffer up to prefix length
   - Copies page_indices row (reference, not deep copy)
   - Increments ref counts on all fully-used pages
   - Allocates fresh page for partial tail (if any)
   - Sets `clone_sources[child] = parent`
3. **Divergent sampling**: In `_handle_clones()`:
   - Reuses parent's logits from the same position
   - Samples with different PRNG key
   - Updates clone's KV cache independently from this point
4. **Independent progress**: After first divergence, clone proceeds as independent sequence

**Memory efficiency**: Shared pages have `ref_count > 1`. Only divergent suffix uses separate pages.

### Page Allocation Strategy

**Goal**: Minimize memory waste while supporting dynamic batching.

**Key ideas**:
1. **Reference counting**: Pages can be shared (ref_count > 1) for prefix of clones
2. **Lazy allocation**: Pages allocated as needed during `allocate_for_seq()`
3. **Greedy allocation**: `argmin(ref_counts)` picks first free page (ref_count=0)
4. **Batched allocation**: All pages for a batch allocated in single pass (fori_loop)

**Page freeing**:
- When sequence finishes, pages are freed in `free_pages_for_finished()`
- Ref counts decremented, but pages not reused until ref_count=0
- Allows safe sharing during clones

### Token Queue Management

**Purpose**: Decouple sampling from model execution. Sampling produces tokens, queue buffers them, decode consumes them.

**Invariants**:
- `queued_tokens[i]`, `queued_slot_ids[i]`, `queued_pos_ids[i]` are aligned
- `num_queued_tokens` tracks valid entries
- Tokens beyond `num_queued_tokens` are `INVALID`

**Operations**:
- `enqueue_tokens()`: Append to queue at offset `num_queued_tokens`
- `pack_next_sequence()`: Dequeue up to `max_tokens`, roll queue left, invalidate tail
- `purge_queue_of_slot()`: Remove all tokens for a slot (e.g., if prematurely terminated)

**Sorting**: `pack_next_sequence()` sorts by slot_id before returning. This ensures tokens from same sequence are contiguous, enabling efficient paged attention.

### Prefill vs. Decode

**Prefill**:
- Processes prompt tokens (multiple tokens per sequence)
- Allocates initial KV pages
- Samples only at sequence boundaries (last token of each prompt)
- Single invocation of model.decode() for entire batch
- Outputs feed into queue for decode

**Decode**:
- Processes one token per sequence per iteration (autoregressive)
- May allocate additional pages if sequences grow
- Samples at every token (since all are boundaries)
- Iterative loop, each iteration is one "step"
- Continues until all sequences finish or max_rounds reached

**Why separate?**: Prefill benefits from high parallelism (many tokens), decode benefits from low latency (single token per sequence).

### Stop Token Detection

**Stop tokens**: User-specified sequences that trigger early termination (e.g., `["\n\n", "###"]`).

**Storage**: `stop_tokens[seq, stop_seq, position]` (left-padded with INVALID)

**Check**: After each new token sampled:
1. Extract tail window from `tokens[seq]` of length `max_stop_tokens`
2. Compare against all stop sequences for this seq
3. If any match (accounting for padding), set `finished[seq] = True`

**Implementation**: `is_stop_signal()` in utils.py:46

### Logprobs Handling

**When enabled**:
- `DecodeState.logprobs` buffer allocated `[seq, position]`
- Sampler returns `(tokens, log_probs)`
- Log probs stored alongside tokens
- Prefix positions set to `nan` (no logprobs for prompt)
- Extracted and returned to user via `DecodeResult.logprobs`

**Why optional?**: Saves memory and compute when not needed.

---

## Memory Management & Page System

### Why Paging?

Traditional attention caches have two issues:
1. **Fragmentation**: Each sequence allocated contiguous buffer, wasted if seq finishes early
2. **No sharing**: Cannot share prefixes between multiple samples

**Paged solution**: KV cache divided into fixed-size pages (e.g., 128 tokens each). Sequences reference pages via indirection table.

**Benefits**:
- Reduced fragmentation (unused pages returned to pool)
- Efficient prefix sharing (clones increment ref counts)
- Dynamic allocation (pages allocated as sequences grow)

### PageTable Architecture

**Structure**:
- `page_ref_counts[page_id]`: How many sequences reference this page
- `page_size`: Tokens per page (static, typically 128)
- `max_seqs`, `pages_per_seq`: Capacity limits

**Allocation**:
```python
free_page_idx = argmin(page_ref_counts)  # Find page with ref_count=0
page_ref_counts[free_page_idx] += 1
page_indices[seq_id, page_idx] = free_page_idx
```

**Freeing**:
```python
for page in page_indices[seq_id]:
    page_ref_counts[page] -= 1  # Decrement ref count
page_indices[seq_id] = INVALID  # Clear mapping
```

### KV Cache Layout

The actual KV cache structure depends on the model (e.g., for transformers with multi-head attention):
```python
cache: list[tuple[NamedArray, NamedArray]]  # One per layer
# Each tuple: (keys, values)
# Shape: [num_pages, num_heads, page_size, head_dim]
```

**Paging**: Given `page_id` and `offset`, the KV entry is at:
```python
cache[layer][0][page_id, head, offset, :]  # Keys
cache[layer][1][page_id, head, offset, :]  # Values
```

**PageBatchInfo**: Tells model where to read/write:
- `new_token_dests[i]`: Flattened index for token i's KV entry
- Model computes: `page_id = dest // page_size`, `offset = dest % page_size`

### Memory Budget Calculation

**Function**: `InferenceEngine._infer_max_pages_from_hbm()` (engine.py:1370)

Uses binary search to find max pages that fit in HBM:

```python
budget = hbm_utilization * free_hbm  # e.g., 0.9 * 40GB

def cache_bytes(num_pages):
    table = PageTable.init(num_pages, max_seqs, page_size, pages_per_seq)
    cache_shape = model.initial_cache(table.spec())
    return tree_byte_size(cache_shape)

# Binary search
low, high = 1, initial_guess
while low + 1 < high:
    mid = (low + high) // 2
    if cache_bytes(mid) <= budget:
        low = mid
    else:
        high = mid

max_pages = low
```

**Result**: Maximizes KV cache capacity while staying within HBM budget.

---

## Performance Considerations

### JIT Compilation

**All hot paths JIT-compiled**:
- `_run_prefill()`: Prefill kernel
- `_run_generation_loop()`: Decode loop
- `DecodeState.allocate_for_seq()`: Page allocation
- `SequenceTable` methods: Page management
- `TokenQueue.pack_next_sequence()`: Queue operations

**Benefits**:
- Kernel fusion (eliminate intermediate materialization)
- Efficient lowering to XLA HLO
- Optimal use of accelerator memory hierarchy

**Trade-off**: First call has compilation overhead, subsequent calls fast.

### Continuous Batching

**Goal**: Maximize accelerator utilization by admitting new requests as others finish.

**Implementation**:
- After each generation iteration, call `_admit_from_queue()`
- If free slots and pages available, admit more requests
- New requests processed via prefill, then join decode loop
- No need to wait for batch to fully complete before starting next

**Benefit**: Higher throughput, lower average latency.

### Iteration Granularity

**`max_rounds`**: Limits iterations per JIT invocation.

**Trade-off**:
- **Higher max_rounds**: More throughput (fewer host↔device syncs), higher latency per iteration
- **Lower max_rounds**: Lower latency (more frequent output extraction), more overhead

**Typical**: `max_rounds=32` balances throughput and responsiveness.

### Batching Strategy

**`max_tokens_per_round`**: How many tokens to pack into each decode iteration.

**Packing efficiency**: Higher when sequences have similar lengths (less padding).

**Admission control**: `_admit_from_queue()` simulates resource usage to avoid OOM.

### Memory-Compute Overlap

**Async operations**: JAX's async dispatch allows host to prepare next batch while device executes current batch.

**In practice**:
- Device executes `_run_generation_loop()`
- Host extracts outputs, admits new requests, prepares prefill work
- Overlapping reduces wall-clock latency

---

## Subtle Behaviors & Implementation Details

### Segment Operations & Dense IDs

**Why segment operations?** In `allocate_for_seq()` (jit_scheduler.py:211), token_slot_ids aren't necessarily sorted or contiguous. A batch might have tokens like `[0, 0, 2, 2, 0, 2]`.

**The problem**: `jax.ops.segment_sum` requires dense segment IDs (0, 1, 2, ...) without gaps.

**The solution**: `get_unique_in_order()` (utils.py:127):
```python
# Input: [0, 0, 2, 2, 0, 2]
unique_ids, dense_ids = get_unique_in_order(slot_ids, size=max_seqs+1, fill_value=INVALID)
# unique_ids: [0, 2, INVALID, ...]  (unique values in order of first appearance)
# dense_ids: [0, 0, 1, 1, 0, 1]  (remapped to dense 0-indexed values)

# Now we can use segment_sum:
segment_lengths = jax.ops.segment_sum(
    data=jnp.ones_like(slot_ids),
    segment_ids=dense_ids,
    num_segments=max_seqs
)
# segment_lengths[0] = 3 (slot 0 appears 3 times)
# segment_lengths[1] = 3 (slot 2 appears 3 times, but mapped to dense_id=1)
```

This is critical because the segment operations need contiguous indices but we want to preserve the original sequence IDs in the output.

### The INVALID Sentinel

**Value**: `INVALID = 2_000_000` (utils.py:12)

**Why this value?**
- Larger than any realistic token ID (vocab sizes typically < 100k)
- Larger than any sequence index (max_seqs typically < 1000)
- Larger than any position index (max positions typically < 100k)
- Small enough to not cause overflow issues in int32 arithmetic
- Easily recognizable in debug output

**Usage pattern**: Invalid values are filtered using `is_valid(x) = (x >= 0) & (x != INVALID)` throughout the codebase.

### Stop Token Left-Padding

**Storage**: `stop_tokens[seq, stop_seq, position]` is **left-padded** with INVALID (jit_scheduler.py:573).

**Example**:
```python
# User provides: ["hello", "bye"]
# Tokenized: [104, 101, 108, 108, 111] and [98, 121, 101]
# With max_stop_tokens=8:
stop_tokens[seq, 0] = [INVALID, INVALID, INVALID, 104, 101, 108, 108, 111]
stop_tokens[seq, 1] = [INVALID, INVALID, INVALID, INVALID, INVALID, 98, 121, 101]
```

**Why left-padding?** Makes comparison efficient! Check logic (jit_scheduler.py:899):
```python
# Extract tail of sequence with same length as stop_tokens
row = tokens[seq].array  # e.g., [..., 98, 121, 101]
padded = jnp.concatenate([jnp.full((stop_len,), INVALID), row])
tail = jax.lax.dynamic_slice(padded, (pos + 1,), (stop_len,))

# Now tail and stop_tokens[seq] have same shape and alignment
# Can compare directly: match when valid tokens match
match = is_stop_signal(tail, stop_tokens[seq])
```

Right-padding would require knowing the valid length of each stop sequence and doing complex masking.

### PRNG Key Management

**Per-sequence base key**: `prng_keys[seq_id]` stored in DecodeState (jit_scheduler.py:577)

**Per-position key derivation** (jit_scheduler.py:659):
```python
def prng_key_for(self, slot_id: int, pos_id: int):
    per_pos_key = self.prng_keys[slot_id]
    return jax.random.fold_in(per_pos_key, pos_id)
```

**Why fold_in?** Ensures:
1. **Determinism**: Same (slot, position) always produces same key
2. **Independence**: Different positions get uncorrelated keys
3. **No state mutation**: Pure functional style, no RNG state to track

**Clone keys**: When creating clones (engine.py:1010):
```python
child_params = dataclasses.replace(
    seq_params,
    key=jax.random.fold_in(seq_params.key, k)  # k is the clone index
)
```

This ensures clones sample differently from parents even at the same position.

### Stable Sorting

**In pack_next_sequence()** (jit_scheduler.py:1090):
```python
slot_ids_sort_order = jnp.argsort(slot_ids.array, stable=True)
```

**Why stable=True?** Preserves position order within each slot ID group:
- Input: `slot_ids=[2, 0, 2, 0]`, `pos_ids=[0, 0, 1, 1]`
- After stable sort: `slot_ids=[0, 0, 2, 2]`, `pos_ids=[0, 1, 0, 1]`
- Without stable: might be `pos_ids=[1, 0, 1, 0]` (wrong order!)

This ensures tokens are in correct temporal order for each sequence, critical for causal attention.

### The Purge Operation

**Used to remove finished sequences** (utils.py:92):
```python
def purge(array: NamedArray, mask: NamedArray, invalid=INVALID):
    # mask=True means "remove this element"
    indices = jnp.nonzero(~mask, size=max_nnz, fill_value=INVALID)[0]
    new_values = array.at[indices].get(mode='fill', fill_value=invalid)
    return new_values
```

**Example**:
```python
tokens = [10, 20, 30, 40]
mask = [False, True, False, True]  # Remove indices 1 and 3

indices = jnp.nonzero([True, False, True, False], size=4)[0]
# indices = [0, 2, INVALID, INVALID]

new_values = tokens[indices]  # [10, 30, INVALID, INVALID]
```

This compacts the array while staying JIT-safe (no dynamic reshaping).

### Cumulative Query Lengths

**Field**: `cu_q_lens` in PageBatchInfo (page_table.py:72)

**Purpose**: Enables efficient packed/concatenated attention (like Flash Attention):
```python
# Example batch with 3 sequences of lengths [4, 3, 2]
cu_q_lens = [0, 4, 7, 9]

# For sequence i:
start = cu_q_lens[i]
end = cu_q_lens[i+1]
seq_tokens = packed_tokens[start:end]
```

**Computation** (jit_scheduler.py:261):
```python
cu_new_counts = hax.concatenate(
    "seq",
    [
        hax.zeros({"seq": 1}),  # Start with 0
        hax.cumsum(new_counts, "seq")  # Cumulative sum of lengths
    ]
)
# new_counts = [4, 3, 2]
# cu_new_counts = [0, 4, 7, 9]
```

Used by attention kernels to know where each sequence's keys/values are in the packed cache.

### Masked Set Operation

**Used throughout for JIT-safe partial updates** (utils.py:27):
```python
def masked_set(dest, axis, start, src, num_to_copy):
    # Copy src[:num_to_copy] into dest[start:start+num_to_copy]
    # But JIT-safe: num_to_copy can be dynamic

    src_arange = hax.arange(src.resolve_axis(axis))
    dest_arange = hax.where(src_arange >= num_to_copy, dest_axis_size, src_arange + start)
    src_arange = hax.where(src_arange >= num_to_copy, src_arange.size, src_arange)

    return dest.at[{axis: dest_arange}].set(src[axis, src_arange], mode="drop")
```

**Trick**: Out-of-bounds indices automatically dropped due to `mode="drop"`, making it safe even when `num_to_copy < src.size`.

### Dual Page Tracking: kv_pages vs page_indices

**Both fields exist in SequenceTable** (jit_scheduler.py:79-80):
- `kv_pages[seq, page]`: Physical page assignments
- `page_indices[seq, page]`: Same values, used during allocation

**Why duplicate?** Historical artifact - they're kept in sync:
```python
# After allocation (jit_scheduler.py:356):
kv_pages = self.kv_pages.at["seq", safe_updated].set(page_indices["seq", safe_updated])
```

**In practice**: `page_indices` is the "working" array during allocation, `kv_pages` is the "stable" view. Could potentially be unified in a refactor.

### Error Handling with eqx.error_if

**Pattern** (jit_scheduler.py:313):
```python
has_free = hax.any(ref_counts == 0).scalar()
ref_counts = eqx.error_if(ref_counts, ~has_free, "Out of free pages")
```

**How it works**: In JIT code, exceptions can't be raised normally. `eqx.error_if`:
1. Takes a value to return
2. Takes a condition
3. If condition is True at runtime, raises error
4. At compile time, compiles both branches

This allows "assertion-like" checks in JIT code while keeping it differentiable.

### Clone Discharge Mechanism

**After first sampling in prefill**, clones are "discharged" (jit_scheduler.py:728):
```python
def discharge_clone(self, target_slot_ids, num_targets):
    # Set clone_sources[target_slot_ids[:num_targets]] = INVALID
    # This marks: "clone has been processed, don't re-sample"
```

**Why?** Prevents clones from being sampled multiple times in the same prefill batch. After first divergence, clones become independent sequences and progress normally through the decode loop.

### Generation Loop Exit Conditions

**The while_loop can exit for three reasons** (engine.py:603):
```python
def cond(state):
    return (
        (step < max_rounds) &                          # 1. Round limit
        (num_queued_tokens > 0) &                      # 2. Queue not empty
        (~hax.all(decode_state.finished))              # 3. Not all finished
    )
```

**Subtle**: If queue becomes empty mid-generation (shouldn't happen with correct logic, but possible in edge cases), the loop exits even if sequences aren't finished. The host-side loop in `generate()` will call `_run_generation_loop()` again if needed.

### Finished Flag Monotonicity

**Once a sequence is marked finished, it stays finished** (jit_scheduler.py:1199):
```python
# In _DecodeOutputs.append():
new_finished = self.finished | finished_snapshot
```

**Why monotonic?** Simplifies logic - no need to "un-finish" a sequence. Once done, its pages can be freed and slot reused.

### Admission Control Constraints

**Four constraints checked** (engine.py:865):
```python
if (
    sim_slots < need_slots or              # 1. Enough free sequence slots?
    sim_pages < need_pages or              # 2. Enough free KV pages?
    sim_tokens + len(prompt) > max_prefill_size or  # 3. Prefill buffer size?
    primaries_in_batch >= max_seqs_in_prefill       # 4. Primary seqs in batch?
):
    break
```

**Constraint 4 is subtle**: `max_seqs_in_prefill` limits the number of **primary** sequences (not total including clones) in a single prefill batch. This is because:
- Prefill needs to sample at boundaries → output buffer sized for `max_seqs_in_prefill` samples
- Clones handled separately in `_handle_clones()`, don't count against this limit
- Prevents prefill batch from growing too large in terms of sampling work

### Thread Safety in openai.py

**Two-thread architecture** (openai.py:204):
1. **Inference thread** (`_inference_loop`): Collects requests into batches
2. **Batch thread** (`_batch_processing_loop`): Executes batches on device

**Thread coordination**:
- `request_queue`: Feeds from async handlers to inference thread
- `batch_queue`: Feeds from inference thread to batch thread
- `model_lock`: Protects model/engine during weight reloading

**Reload logic** (openai.py:241):
```python
def reload(self, weight_callback):
    with self.model_lock:  # Block new batches
        # Wait for current batch to finish (automatic via lock)
        self.model = weight_callback(self.model)
        self.engine = InferenceEngine.from_model_with_config(...)
        # New batches can now proceed with updated model
```

This ensures hot-swapping of weights without interrupting in-flight requests.

### Batch Timeout for Continuous Batching

**In _inference_loop** (openai.py:302):
```python
requests = _fetch_all_from_queue(self.request_queue, self.config.batch_timeout)
# batch_timeout typically 0.1s
```

**Trade-off**:
- **Higher timeout**: Larger batches (better throughput), higher latency for first request
- **Lower timeout**: Smaller batches (worse throughput), lower latency

**Typical**: 100ms balances batching opportunity with responsiveness.

---

## Summary

A batch's journey through Levanter inference:

1. **User submits requests** → Enqueued to `request_queue`
2. **Admission control** → Batch requests that fit in memory
3. **Prefill packing** → Host builds `PrefillWork` with prompt tokens and clone instructions
4. **Device prefill** → Allocate slots, pages, run model on prompts, sample first tokens, handle clones
5. **Output extraction** → Transfer outputs to host, update results
6. **Decode loop** → Iteratively pack queued tokens, run model, sample, enqueue, until all finish
7. **Continuous batching** → Admit new requests mid-generation if capacity allows
8. **Finalization** → Assemble per-request outputs, free resources, return results

**Key innovations**:
- **Paged attention**: Efficient KV cache with sharing and dynamic allocation
- **Clone-based multi-sampling**: Share prefix computation, diverge only where necessary
- **JIT-compiled state machine**: All hot paths optimized for accelerators
- **Continuous batching**: Maximize utilization by overlapping request lifetimes

This architecture achieves high throughput and low latency for serving language models at scale.
