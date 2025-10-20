# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from levanter.inference.decode_state import (
    DecodeState,
    KvPageCache,
    PageCache,
    PageTableSpec,
    SeqDecodingParams,
)
from levanter.inference.utils import INVALID
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from levanter.utils.jax_utils import estimated_free_device_memory, sharded_tree_size

logger = logging.getLogger(__name__)


def _tree_byte_size(tree) -> int:
    """Return the per-device number of bytes represented by ``tree``."""

    return sharded_tree_size(tree)


def _available_hbm_budget_bytes(hbm_utilization: float) -> int:
    """Estimate the per-device HBM budget available to the KV cache."""

    if not (0.0 < hbm_utilization <= 1.0):
        raise ValueError("hbm_utilization must be in the interval (0, 1].")

    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available for inference.")

    budgets: list[int] = []
    bytes_per_gib = 1024**3
    for device in devices:
        free_gib = estimated_free_device_memory(device)
        if free_gib is None:
            raise RuntimeError(f"Device {device} does not expose memory statistics.")
        free_bytes = max(int(free_gib * bytes_per_gib), 0)
        budgets.append(int(free_bytes * hbm_utilization))

    if not budgets:
        raise RuntimeError("Unable to determine device HBM budget.")

    return min(budgets)


@dataclass(frozen=True)
class Request:
    """A request for generation of a single sequence."""

    prompt_tokens: list[int]
    request_id: int
    decode_params: SeqDecodingParams
    n_generations: int
    enable_logprobs: bool = False


@dataclasses.dataclass
class DecodeResult:
    """Holds per-(request, choice) decode outputs and status."""

    id: int
    choice: int
    token_list: list[int]
    # Count of newly appended tokens (includes prompt tokens as extracted)
    tokens_decoded: int = 0
    done: bool = False
    logprobs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class InferenceEngineConfig:
    """Configuration for Engine memory/layout knobs.

    Exposes key buffer sizes and limits controlling prefill, decode queueing, and page table capacity.
    """

    max_seq_len: int
    """
    Maximum sequence length (including prompt). Used for validation and buffer sizing at init.
    """

    hbm_utilization: float = 0.9
    """Fraction of device HBM to reserve for the KV cache when :attr:`max_pages` is ``None``."""

    page_size: int = 128
    """Tokens per KV page."""

    max_rounds: int = 32
    """Maximum number of while-loop iterations per decode call. Higher values increase throughput but also latency."""

    # Stop-token capacity (used for validation and buffer sizing at init)
    max_stop_seqs: int = 4
    """Maximum number of stop sequences per active sequence. 0 disables stop tokens."""
    max_stop_tokens: int = 16
    """Maximum tokens per stop sequence (position axis length)."""

    # Default PRNG seed for building per-request keys (optional convenience)
    seed: int = 0

    enable_logprobs: bool = False
    """Enable computing logprobs for generated tokens."""

    # You probably don't need to tune the knobs below

    max_seqs: int = 256
    """Maximum concurrent sequences (local slots)."""

    max_pages: Optional[int] = None
    """Total number of KV pages available. If None, inferred from :attr:`hbm_utilization`."""

    compute_dtype: jnp.dtype = jnp.bfloat16
    """KV cache dtype. Default bfloat16 for performance/accuracy balance."""

    max_queued_tokens: int = 512
    """Capacity of the token queue used between sampling and decode packing."""

    max_seqs_in_prefill: int = 16
    """Maximum number of sequences to batch in prefill before flushing."""

    # Decode loop knobs
    max_tokens_per_round: int | None = None
    """Pack size for each decode loop iteration. If None, set to max_seqs """

    def __post_init__(self):
        # this one is only required because of clones. If we really care, we could relax this
        if self.max_queued_tokens < self.max_seqs:
            raise ValueError("max_queued_tokens must be >= max_seqs")

        if self.max_queued_tokens < self.imputed_max_tokens_per_round:
            raise ValueError("max_queued_tokens must be >= max_tokens_per_round")

        if self.max_queued_tokens < self.max_seqs_in_prefill:
            raise ValueError("max_queued_tokens must be >= max_seqs_in_prefill")

    @property
    def imputed_max_tokens_per_round(self) -> int:
        """Return explicit `max_tokens_per_round` or default to `max_seqs` when unset."""
        return self.max_tokens_per_round if self.max_tokens_per_round is not None else self.max_seqs

    @property
    def max_pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size


def _infer_max_pages_from_hbm(model: LmHeadModel, config: InferenceEngineConfig) -> int:
    """Infer a KV-page budget using HBM utilization targets."""

    max_pages_per_seq = config.max_pages_per_seq

    try:
        budget = _available_hbm_budget_bytes(config.hbm_utilization)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        logger.warning(
            "Falling back to max_seqs * max_pages_per_seq for KV cache sizing because HBM budget "
            "could not be determined: %s",
            exc,
        )
        return int(config.max_seqs * max_pages_per_seq)

    @functools.lru_cache(maxsize=None)
    def cache_bytes(num_pages: int) -> int:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive when sizing the KV cache.")

        def initial_cache(pages: int) -> int:
            table = PageTable.init(pages, config.max_seqs, config.page_size, max_pages_per_seq)
            cache_shape = model.initial_cache(table.spec(), dtype=config.compute_dtype)
            return cache_shape

        cache_shape = eqx.filter_eval_shape(initial_cache, num_pages)

        return _tree_byte_size(cache_shape)

    bytes_one = cache_bytes(1)
    if bytes_one > budget:
        raise ValueError(
            "HBM budget insufficient to allocate even a single KV cache page. "
            "Provide `max_pages` explicitly or increase `hbm_utilization`."
        )

    # Use the previous heuristic as the initial guess before expanding.
    guess = max(int(config.max_seqs * max_pages_per_seq), 1)

    low = 1
    high = guess
    high_bytes = cache_bytes(high)

    if high_bytes <= budget:
        low = high
        while True:
            high *= 2
            if high > (1 << 20):
                warnings.warn(
                    "KV cache size exceeded 1M pages during budget inference; "
                    "aborting search and using current estimate."
                )
                high = 1 << 20
                break
            high_bytes = cache_bytes(high)
            if high_bytes > budget:
                break
            low = high

    # Binary search between the known-good lower bound and the first oversized bound.
    while low + 1 < high:
        mid = (low + high) // 2
        mid_bytes = cache_bytes(mid)
        if mid_bytes <= budget:
            low = mid
        else:
            high = mid

    max_pages = low

    bytes_at_max = cache_bytes(max_pages)
    next_bytes = cache_bytes(high)
    per_page = bytes_at_max if max_pages == 1 else bytes_at_max - cache_bytes(max_pages - 1)
    base_bytes = max(bytes_at_max - per_page * max_pages, 0)

    import humanfriendly as hly

    logger.info(
        "Auto-computed KV cache budget: base=%s, per_page=%s, budget=%s, used=%s, next=%s -> max_pages=%d",
        hly.format_size(base_bytes),
        hly.format_size(per_page),
        hly.format_size(budget),
        hly.format_size(bytes_at_max),
        hly.format_size(next_bytes),
        max_pages,
    )

    return max_pages


def _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices):
    """
    Compute positions of last tokens per sequence inside a packed slice.

    Boundary when absolute pos_id equals the post-allocation seq_len - 1 for that sequence.
    """
    seq_lens_per_seq = seq_lens["seq", slot_ids]
    boundary_mask = pos_ids == (seq_lens_per_seq - 1)
    # jax.debug.print("pos_ids={pos} seq_lens={lens} boundary={b}", pos=pos_ids.array, lens=seq_lens_per_seq.array, b=boundary_mask.array)
    sample_indices = hax.where(
        boundary_mask,
        fill_value=INVALID,
        new_axis=pos_ids.resolve_axis("position").resize(max_sample_indices),
    )[0]
    return sample_indices


def _check_stop_sequences(
    generated_tokens: np.ndarray,  # [num_seqs, num_generated]
    stop_tokens: list[list[int]],  # per-sequence stop token lists
    max_stop_len: int,
) -> np.ndarray:  # [num_seqs] bool array
    """Check if any sequence has generated a stop token.

    Args:
        generated_tokens: Array of shape [num_seqs, num_generated] containing all generated tokens so far
        stop_tokens: List of stop token lists per sequence
        max_stop_len: Maximum length of stop sequences to check

    Returns:
        Boolean array of shape [num_seqs] indicating which sequences are finished
    """
    num_seqs = generated_tokens.shape[0]
    finished = np.zeros(num_seqs, dtype=bool)

    for seq_idx in range(num_seqs):
        if not stop_tokens[seq_idx]:
            continue

        # Get tokens for this sequence
        seq_tokens = generated_tokens[seq_idx]

        # Check if any stop token appears in the generated tokens
        for stop_token in stop_tokens[seq_idx]:
            # Simple single-token check
            if np.any(seq_tokens == stop_token):
                finished[seq_idx] = True
                break

    return finished


@functools.partial(jax.jit, donate_argnums=0)
def _run_prefill(
    cache: PageCache,
    decode_state: DecodeState,
    model: LlamaLMHeadModel,
) -> PageCache:
    """Run prefill using a fresh, local token queue. Only populates the KV-cache."""
    batch_info = decode_state.batch_info(inner_iteration=0, kv_cache=cache)
    _, cache = model.decode(
        input_ids=batch_info.tokens,
        kv_cache=cache,
        batch_info=batch_info,
        pos_ids=batch_info.pos_ids,
    )
    return cache


class DecodeOutputs(eqx.Module):
    tokens: jnp.ndarray
    logprobs: jnp.ndarray

    def update(self, new_tokens: hax.NamedArray, new_logprobs: hax.NamedArray, step: jnp.ndarray) -> "DecodeOutputs":
        return DecodeOutputs(
            tokens=lax.dynamic_update_slice(self.tokens, jnp.expand_dims(new_tokens.array, 1), (0, step)),
            logprobs=lax.dynamic_update_slice(self.logprobs, jnp.expand_dims(new_logprobs.array, 1), (0, step)),
        )


class DecodeLoopState(eqx.Module):
    step: jnp.ndarray
    cache: PageCache
    decode_state: DecodeState
    outputs: DecodeOutputs


# @hax.named_jit(donate_args=(True, False, False))
@functools.partial(jax.jit, static_argnums=(3, 4), donate_argnames=("page_cache",))
def _run_generation_loop(
    page_cache: PageCache,
    decode_state: DecodeState,
    model: LlamaLMHeadModel,
    sampler: Sampler,
    num_rounds: int,
) -> tuple[PageCache, DecodeOutputs, DecodeState]:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached."""

    def body(state: DecodeLoopState) -> DecodeLoopState:
        jax.debug.print("[DECODE_LOOP] step={s} offset={o}", s=state.step, o=state.decode_state.offset)

        binfo = state.decode_state.batch_info(inner_iteration=state.step, kv_cache=state.cache)

        jax.debug.print(
            "[BATCH_INFO] tokens={t} pos_ids={p} seq_lens={sl}",
            t=binfo.tokens.array,
            p=binfo.pos_ids.array,
            sl=binfo.seq_lens.array,
        )
        jax.debug.print(
            "[BATCH_INFO] cu_q_lens={c} new_token_dests={d}", c=binfo.cu_q_lens.array, d=binfo.new_token_dests.array
        )

        # Decode logits and sample new tokens
        logits, cache = model.decode(binfo.tokens, state.cache, binfo, binfo.pos_ids)

        seed = jax.random.PRNGKey(state.step)
        seed = jnp.tile(seed, binfo.num_seqs)
        seed = seed.reshape(binfo.num_seqs, 2)
        prng_keys = jax.vmap(jax.random.fold_in)(seed, binfo.pos_ids.array)
        temps = 0.7  # state.decode_state.temperature["seq", new_slot_ids]
        new_tokens, logprobs = hax.vmap(sampler, "position")(logits, temps, key=prng_keys)

        jax.debug.print("[SAMPLED] new_tokens={t} logprobs={lp}", t=new_tokens.array, lp=logprobs.array)

        # Update decode state with the sampled tokens
        decode_state = state.decode_state.update_tokens(new_tokens=new_tokens, new_logprobs=logprobs, step=state.step)
        outputs = state.outputs.update(new_tokens=new_tokens, new_logprobs=logprobs, step=state.step)

        jax.debug.print(
            "[AFTER_UPDATE] tokens={t} pos_ids={p} seq_lens={sl}",
            t=decode_state.tokens.array,
            p=decode_state.pos_ids.array,
            sl=decode_state.seq_lens.array,
        )

        return DecodeLoopState(
            step=state.step + 1,
            cache=cache,
            outputs=outputs,
            decode_state=decode_state,
        )

    def cond(state):
        return state.step < num_rounds

    # Allocate an outputs buffer sized for this run
    outputs_buf = DecodeOutputs(
        logprobs=jnp.zeros((decode_state.num_seqs, num_rounds), dtype=jnp.float32),
        tokens=jnp.zeros((decode_state.num_seqs, num_rounds), dtype=jnp.int32),
    )
    init_state = DecodeLoopState(step=0, cache=page_cache, outputs=outputs_buf, decode_state=decode_state)
    final_state = jax.lax.while_loop(cond, body, init_state)
    return final_state.cache, final_state.outputs, final_state.decode_state


@dataclass
class GenerationResult:
    tokens: list[list[int]]
    logprobs: list[list[float]] | None
    total_generated: int


class InferenceEngine:
    """Encapsulates batch inference: prefill + decode + output extraction.

    Typical usage:

        svc = Engine.from_model(model, tokenizer, Vocab, max_seqs, max_pages, page_size, max_pages_per_seq, compute_dtype)
        texts = svc.generate(requests)
    """

    config: InferenceEngineConfig
    model: LmHeadModel
    cache: KvPageCache
    tokenizer: Any
    sampler: Sampler
    page_spec: PageTableSpec

    def __init__(
        self,
        *,
        model: LmHeadModel,
        tokenizer,
        cache: KvPageCache,
        sampler: Sampler,
        config: InferenceEngineConfig,
        page_spec: PageTableSpec,
    ) -> None:
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.config = config
        self.page_spec = page_spec

    @classmethod
    def from_model_with_config(
        cls,
        model: LlamaLMHeadModel,
        tokenizer,
        config: InferenceEngineConfig,
    ) -> "InferenceEngine":
        """Build an engine using a EngineConfig for sizing knobs."""
        if config.max_pages is None:
            inferred_pages = _infer_max_pages_from_hbm(model, config)
            config = dataclasses.replace(config, max_pages=int(inferred_pages))

        assert config.max_pages is not None

        spec = PageTableSpec(config.max_pages, config.page_size, config.max_seqs)
        cache = hax.named_jit(model.initial_cache)(spec, dtype=config.compute_dtype)
        vocab_axis = model.Vocab
        sampler = Sampler(vocab_axis)
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            sampler=sampler,
            config=config,
            page_spec=spec,
        )

    def generate(self, requests: Sequence[Request], step_callback=None) -> GenerationResult:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).

        Args:
            requests: Sequence of generation requests
            step_callback: Optional callback function called at each decode iteration with iteration number
        """
        # validate we don't have any sequences with n_generations exceeding max_seqs
        max_needed = max(int(r.n_generations) for r in requests)
        if max_needed > int(self.page_spec.max_seqs):
            raise ValueError(
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.page_spec.max_seqs})."
                "Decompose your request into smaller batches or increase max_seqs when building the service."
            )

        # construct the DecodeState inputs for the batch of requests, run prefill, then prepare the clones
        # N.B. Llama decode doesn't support the "batch" axis. We're therefore forced
        # to pack all sequences into linear arrays.
        seq_lens = []
        tokens = []
        pos_ids = []
        cu_q_lens = []

        q_len_offset = 0
        total_len = sum([len(req.prompt_tokens) for req in requests])
        num_seqs = len(requests)
        total_pages = num_seqs * self.page_spec.pages_per_seq
        page_indices = np.arange(total_pages).reshape((num_seqs, self.page_spec.pages_per_seq))

        for i, req in enumerate(requests):
            seq_lens.append(len(req.prompt_tokens))
            tokens.extend(req.prompt_tokens)
            pos_ids.extend(np.arange(len(req.prompt_tokens)))
            cu_q_lens.append(q_len_offset)
            q_len_offset += len(req.prompt_tokens)

        # Add terminal element to cu_q_lens (required by ragged paged attention kernel)
        cu_q_lens.append(q_len_offset)

        # now run prefill. this will fill the KV cache for all of the sequences. we'll then
        # build a new decode state with these pages as the baseline, and new page indices for
        # the newly decoded tokens.
        decode_state = DecodeState(
            page_spec=self.page_spec,
            seq_lens=hax.NamedArray(np.array(seq_lens), {"seq": num_seqs}),
            tokens=hax.NamedArray(np.array(tokens), {"position": total_len}),
            pos_ids=hax.NamedArray(np.array(pos_ids), {"position": total_len}),
            cu_q_lens=hax.named(np.array(cu_q_lens, dtype=np.int32), "seq"),  # size is num_seqs + 1
            page_indices=hax.NamedArray(page_indices, {"seq": num_seqs, "page": self.page_spec.pages_per_seq}),
            logprobs=hax.zeros({"position": total_len}),
            offset=0,
            finished=hax.zeros({"seq": num_seqs}, dtype=jnp.bool_),
        )

        self.cache = _run_prefill(self.cache, decode_state, self.model)

        # The prefill kernel has now populated the KV cache for our initial set
        # of prompts. For clones (n_generations > 1), we populate the
        # `page_indices` of the clone for the first N tokens to refer to the
        # parent sequence.
        # TODO(prefill expansion)

        seq_lens = []
        tokens = []
        pos_ids = []
        cu_q_lens = []

        q_len_offset = 0
        total_len = sum([len(req.prompt_tokens) for req in requests])
        num_seqs = len(requests)
        total_pages = num_seqs * self.page_spec.pages_per_seq
        page_indices = np.arange(total_pages).reshape((num_seqs, self.page_spec.pages_per_seq))

        for i, req in enumerate(requests):
            seq_lens.append(len(req.prompt_tokens))
            tokens.append(req.prompt_tokens[-1])
            pos_ids.append(len(req.prompt_tokens) - 1)  # Position of last prompt token
            cu_q_lens.append(i)  # Offset in current batch (0, 1, 2, ...)

        # Add terminal element to cu_q_lens (required by ragged paged attention kernel)
        cu_q_lens.append(num_seqs)

        decode_state = DecodeState(
            page_spec=self.page_spec,
            seq_lens=hax.NamedArray(np.array(seq_lens), {"seq": num_seqs}),
            tokens=hax.NamedArray(np.array(tokens), {"position": num_seqs}),
            pos_ids=hax.NamedArray(np.array(pos_ids), {"position": num_seqs}),
            cu_q_lens=hax.named(np.array(cu_q_lens, dtype=np.int32), "seq"),  # size is num_seqs + 1
            page_indices=hax.NamedArray(page_indices, {"seq": num_seqs, "page": self.page_spec.pages_per_seq}),
            logprobs=hax.zeros({"position": num_seqs}),
            offset=0,
            finished=hax.zeros({"seq": num_seqs}, dtype=jnp.bool_),
        )

        # Extract stop tokens for each request
        stop_tokens_per_seq = []
        for req in requests:
            if req.decode_params.stop_tokens is not None:
                # Extract stop tokens from SeqDecodingParams
                stop_tok = req.decode_params.stop_tokens
                if isinstance(stop_tok, hax.NamedArray):
                    stop_tokens_per_seq.append(stop_tok.array.tolist())
                else:
                    stop_tokens_per_seq.append(stop_tok)
            else:
                stop_tokens_per_seq.append([])

        # Outer generation loop: run until all sequences finish or we hit max length
        max_outer_rounds = (self.config.max_seq_len + self.config.max_rounds - 1) // self.config.max_rounds
        all_tokens = []
        all_logprobs = []
        finished_mask = np.zeros(num_seqs, dtype=bool)

        for outer_round in range(max_outer_rounds):
            logger.info(f"[OUTER_LOOP] Starting outer_round={outer_round}, offset={decode_state.offset}")

            # Run one batch of generation (inner loop)
            self.cache, decoded_outputs, decode_state = _run_generation_loop(
                page_cache=self.cache,
                decode_state=decode_state,
                sampler=self.sampler,
                model=self.model,
                num_rounds=self.config.max_rounds,
            )

            # Transfer outputs to host
            outputs = jax.device_get(decoded_outputs)
            all_tokens.append(outputs.tokens)
            all_logprobs.append(outputs.logprobs)

            logger.info(f"[OUTER_LOOP] Generated tokens: {outputs.tokens}")

            # Check for stop sequences
            # Concatenate all tokens generated so far for each sequence
            tokens_so_far = np.concatenate([t for t in all_tokens], axis=1)
            finished_mask = _check_stop_sequences(tokens_so_far, stop_tokens_per_seq, self.config.max_stop_tokens)

            # Check if all sequences are done
            if np.all(finished_mask):
                logger.info(f"All sequences finished after {outer_round + 1} outer rounds")
                break

            # Update decode_state for next round
            # Increment offset so next round knows where to write KV cache
            decode_state = dataclasses.replace(
                decode_state,
                offset=decode_state.offset + self.config.max_rounds,
            )

            logger.info(f"[OUTER_LOOP] Updated offset to {decode_state.offset}")

        # Aggregate all tokens across rounds
        all_tokens_concat = np.concatenate(all_tokens, axis=1)  # [num_seqs, total_rounds]
        all_logprobs_concat = np.concatenate(all_logprobs, axis=1)

        # Trim to actual lengths (remove padding zeros or tokens after stop)
        tokens_list = []
        logprobs_list = []
        total_generated = 0

        for i in range(num_seqs):
            seq_tokens = all_tokens_concat[i]
            seq_logprobs = all_logprobs_concat[i]

            # Find where to stop: either at first zero (padding) or when finished
            if finished_mask[i]:
                # Find first occurrence of stop token
                valid_len = len(seq_tokens)
                for stop_tok in stop_tokens_per_seq[i]:
                    first_stop = np.where(seq_tokens == stop_tok)[0]
                    if len(first_stop) > 0:
                        valid_len = min(valid_len, first_stop[0] + 1)  # Include stop token
            else:
                # Not finished, take all non-zero tokens
                nonzero_indices = np.where(seq_tokens != 0)[0]
                valid_len = len(nonzero_indices) if len(nonzero_indices) > 0 else 0

            tokens_list.append(seq_tokens[:valid_len].tolist())
            logprobs_list.append(seq_logprobs[:valid_len].tolist())
            total_generated += valid_len

        return GenerationResult(
            tokens=tokens_list,
            logprobs=logprobs_list,
            total_generated=total_generated,
        )
