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

from levanter.inference import decode_state
from levanter.inference.decode_state import (
    DecodeState,
    GenState,
    KvPageCache,
    PageTableSpec,
    SeqDecodingParams,
)
from levanter.inference.utils import INVALID
from levanter.layers.attention import BatchInfo
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


@functools.partial(jax.jit, donate_argnums=0)
def _prefill_kernel(
    gen_state: GenState,
    model: LlamaLMHeadModel,
) -> GenState:
    """Run prefill using a fresh, local token queue. Only populates the KV-cache."""
    _, cache = model.decode(
        gen_state.decode_state.tokens,
        gen_state.cache,
        gen_state.decode_state.batch_info,
        gen_state.decode_state.pos_ids,
    )
    return dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)


def _run_prefill(
    gen_state: GenState,
    model: LlamaLMHeadModel,
) -> GenState:
    gen_state = _prefill_kernel(gen_state, model)

    # Now iterate over each input sequence, and construct clones which share the parents KV cache
    # pages for each of then `n_generations`.
    for i in range(gen_state.decode_state.num_seqs):
        pass
    return gen_state


# @hax.named_jit(donate_args=(True, False, False))
@functools.partial(jax.jit, static_argnums=(3, 4), donate_argnames=("gen_state",))
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
) -> GenState:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached."""

    def cond(state: tuple[GenState, jax.Array]):
        _gen_state, step = state
        return (
            (step < max_rounds)
            & (_gen_state.decode_state.num_queued_tokens > 0)
            & (~hax.all(_gen_state.decode_state.finished)).scalar()
        )

    def body(state: tuple[GenState, _DecodeOutputs, jax.Array]) -> tuple[GenState, _DecodeOutputs, jax.Array]:
        gen_state, outputs, step = state

        # Pack the next chunk from the queue via DecodeState
        decode_state, packed_seq = gen_state.decode_state.pack_next_sequence(max_tokens_per_round)

        tokens = packed_seq.tokens
        pos_ids = packed_seq.pos_ids
        slot_ids = packed_seq.slot_ids

        # jax.debug.print(
        #     "[_run_gen_loop] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
        #     tokens=tokens.array,
        #     slots=slot_ids.array,
        #     pos=pos_ids.array,
        #     lens=gen_state.decode_state.seq_lens.array,
        # )

        decode_state, binfo = decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)

        seq_lens = decode_state.seq_lens

        max_sample_indices = min(decode_state.page_table.max_seqs, max_tokens_per_round)
        sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices)

        # Decode logits and sample new tokens
        logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
        logits_at_samples = logits["position", sample_indices]

        num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
        new_slot_ids = slot_ids["position", sample_indices]
        new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_slot_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
        # Append non-stateful outputs for host-side extraction
        outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)

        # jax.debug.print(
        #     "[gen] step={step} outputs_size={size} queued_after={queued}",
        #     step=step,
        #     size=outputs.num_tokens,
        #     queued=new_gen_state.decode_state.num_queued_tokens,
        # )
        return new_gen_state, outputs, step + 1

    # Allocate an outputs buffer sized for this run
    outputs_buf = _DecodeOutputs.init(
        max_tokens=max(max_tokens_per_round * max_rounds, 1),
        max_seqs=gen_state.decode_state.max_seqs,
        with_logprobs=True,
    )
    init_state = (gen_state, outputs_buf, jnp.array(0, dtype=jnp.int32))
    final_gen_state, final_outputs, _ = jax.lax.while_loop(cond, body, init_state)
    # jax.debug.print("[gen] final outputs_size={size}", size=final_outputs.num_tokens)
    return final_gen_state, final_outputs


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
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.table.max_seqs})."
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

        for i, req in enumerate(requests):
            seq_lens.append(len(req.prompt_tokens))
            tokens.extend(req.prompt_tokens)
            pos_ids.extend(np.arange(len(req.prompt_tokens)))
            cu_q_lens.append(q_len_offset)
            cu_q_lens[i] = q_len_offset
            q_len_offset += len(req.prompt_tokens)

        token_dests = hax.NamedArray(np.arange(total_len), {"position", }

        # now run prefill. this will fill the KV cache for all of the sequences. we'll then
        # build a new decode state with these pages as the baseline, and new page indices for
        # the newly decoded tokens.
        decode_state = DecodeState.init(
            kv_cache=self.cache,
            page_table=self.page_spec,
            seq_lens=seq_lens,
            tokens=tokens,
            pos_ids=pos_ids,
            cu_q_lens=cu_q_lens,
            batch_info=BatchInfo(
                kv_cache=self.cache,
                page_size=self.page_spec.page_size,
                new_token_dests=token_dests,
                num_seqs=len(requests),
            ),
        )

        gen_state = GenState(self.cache, decode_state)
        gen_state = _run_prefill(gen_state, self.model)
        return GenerationResult(tokens=tokens, logprobs=logprobs, total_generated=sum(seq_lens))
