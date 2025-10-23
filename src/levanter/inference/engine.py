# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
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
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from levanter.utils.jax_utils import estimated_free_device_memory

logger = logging.getLogger(__name__)


def _bucket_size(actual_size: int, buckets: tuple[int, ...]) -> int:
    """Return the smallest bucket >= actual_size."""
    for bucket in buckets:
        if actual_size <= bucket:
            return bucket
    # If exceeds all buckets, round up to next power of 2
    return 1 << (actual_size - 1).bit_length()


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

    tokens_per_round: int = 4
    """Number of while-loop iterations per decode call. Higher values increase throughput but also latency."""

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

    prefill_token_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    """Power-of-2 buckets for prefill token padding. Prefill will pad to next largest bucket."""

    @property
    def max_pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size


def _check_stop_sequences(
    generated_tokens: np.ndarray,
    requests: Sequence[Request],
    final_position: np.ndarray,
) -> np.ndarray:
    """Return the stop index for each for sequence, or -1 if not found."""
    seq_idx = 0

    for req in requests:
        for _ in range(req.n_generations):
            if final_position[seq_idx] != -1:
                seq_idx += 1
                continue
            stop_sequences = req.decode_params.stop_tokens
            if not stop_sequences:
                seq_idx += 1
                continue

            stop_sequences = stop_sequences.array
            tokens = generated_tokens[seq_idx]
            for stop_idx in range(stop_sequences.shape[0]):
                stop_tokens = stop_sequences[stop_idx].tolist()
                for i in range(len(tokens) - len(stop_tokens) + 1):
                    if tokens[i : i + len(stop_tokens)].tolist() == stop_tokens:
                        final_position[seq_idx] = i + len(stop_tokens)
                        break
            seq_idx += 1

    return final_position


def build_prefill_state(
    requests: Sequence[Request],
    page_spec: PageTableSpec,
    token_buckets: tuple[int, ...],
) -> DecodeState:
    """Build decode state for prefill phase with static shapes.

    Pads to static sizes based on power-of-2 bucketing for XLA compilation.
    Padding sequences have seq_lens=0 and are naturally skipped by attention.

    Args:
        requests: Sequence of generation requests
        page_spec: Page table specification
        max_seqs_in_prefill: Maximum number of sequences (for padding)
        token_buckets: Power-of-2 buckets for token padding
        pad_token_id: Token ID to use for padding (default 0)

    Returns:
        DecodeState configured for prefill phase with static shapes
    """
    max_seq_len = _bucket_size(
        max(len(req.prompt_tokens) for req in requests),
        token_buckets,
    )

    seq_lens = np.zeros(page_spec.max_seqs, dtype=np.int32)
    cu_q_lens = np.zeros(page_spec.max_seqs + 1, dtype=np.int32)

    temperature = np.zeros(max_seq_len, dtype=np.float32)
    tokens = np.zeros(max_seq_len, dtype=np.int32)
    pos_ids = np.zeros(max_seq_len, dtype=np.int32)
    tok_offset = 0

    for i, req in enumerate(requests):
        tokens[tok_offset:tok_offset + len(req.prompt_tokens)] = req.prompt_tokens
        pos_ids[tok_offset:tok_offset + len(req.prompt_tokens)] = np.arange(len(req.prompt_tokens))
        temperature[tok_offset:tok_offset + len(req.prompt_tokens)] = req.decode_params.temperature
        seq_lens[i] = len(req.prompt_tokens)
        cu_q_lens[i + 1] = tok_offset + len(req.prompt_tokens)
        tok_offset += len(req.prompt_tokens)

    for i in range(len(requests), page_spec.max_seqs):
        cu_q_lens[i + 1] = cu_q_lens[i]

    token_dests = np.arange(
        page_spec.pages_per_seq * page_spec.page_size * page_spec.max_seqs
    ).reshape(page_spec.max_seqs, -1)

    # Mark finished sequences (padding sequences are already finished)
    finished_array = np.ones(tokens.shape, dtype=bool)
    finished_array[:len(requests)] = False

    # logger.info(
    #     "[PREFILL_STATE] actual_seqs=%s"
    #     "actual_tokens=%s "
    #     "pos_ids=%s "
    #     "seq_lens=%s "
    #     "cu_q_lens=%s ",
    #     len(requests),
    #     tokens,
    #     pos_ids,
    #     seq_lens,
    #     cu_q_lens,
    # )

    return DecodeState(
        num_seqs=len(requests),
        page_spec=page_spec,
        token_dests=hax.named(token_dests, ("seq", "position")),
        seq_lens=hax.named(seq_lens, "seq"),
        temperature=hax.named(temperature, "position"),
        tokens=hax.named(tokens, "position"),
        pos_ids=hax.named(pos_ids, "position"),
        cu_q_lens=hax.named(cu_q_lens, "seq"),
        logprobs=hax.zeros({"position": tokens.shape[0]}),
        finished=hax.named(finished_array, "seq"),
    )


def build_decode_state(
    requests: Sequence[Request],
    page_spec: PageTableSpec,
    prefill_token_dests: np.ndarray,
) -> DecodeState:
    """Build decode state for autoregressive generation with clone support.

    Constructs decode state where each request can have multiple clones (n_generations).
    Clones share prefill KV cache pages but have independent generation state.

    Args:
        requests: Sequence of generation requests
        page_spec: Page table specification
        prefill_token_dests: Token destinations from prefill state

    Returns:
        DecodeState configured for autoregressive decode phase
    """
    seq_lens = np.zeros(page_spec.max_seqs, dtype=np.int32)
    tokens = np.zeros(page_spec.max_seqs, dtype=np.int32)
    pos_ids = np.zeros(page_spec.max_seqs, dtype=np.int32)
    cu_q_lens = np.zeros(page_spec.max_seqs + 1, dtype=np.int32)
    temperature = np.zeros(page_spec.max_seqs, dtype=np.float32)
    num_seqs = sum(int(max(1, req.n_generations)) for req in requests)
    seq_offset = 0

    # Clone token_dests so each clone points to correct prefill pages
    cloned_dests = prefill_token_dests.copy()

    for i, req in enumerate(requests):
        for _ in range(int(req.n_generations)):
            # Share prefill pages across clones
            cloned_dests[seq_offset, : len(req.prompt_tokens)] = prefill_token_dests[i][: len(req.prompt_tokens)]
            seq_lens[seq_offset] = len(req.prompt_tokens)
            tokens[seq_offset] = req.prompt_tokens[-1]  # Last prompt token
            pos_ids[seq_offset] = len(req.prompt_tokens) - 1
            cu_q_lens[seq_offset + 1] = seq_offset + 1
            temperature[seq_offset] = req.decode_params.temperature
            seq_offset += 1
    
    for i in range(num_seqs, page_spec.max_seqs):
        cu_q_lens[i + 1] = cu_q_lens[i]
    
    return DecodeState(
        num_seqs=num_seqs,
        token_dests=hax.named(cloned_dests, ("seq", "position")),
        page_spec=page_spec,
        seq_lens=hax.named(seq_lens, "seq"),
        temperature=hax.named(temperature, "position"),
        tokens=hax.named(tokens, "position"),
        pos_ids=hax.named(pos_ids, "position"),
        cu_q_lens=hax.named(cu_q_lens, "seq"),
        logprobs=hax.zeros({"position": page_spec.max_seqs}),
        finished=hax.zeros({"seq": page_spec.max_seqs}, dtype=jnp.bool_),
    )


@functools.partial(jax.jit, donate_argnums=0)
def _run_prefill(
    cache: PageCache,
    decode_state: DecodeState,
    model: LlamaLMHeadModel,
) -> PageCache:
    """Run prefill using a fresh, local token queue. Only populates the KV-cache."""
    batch_info = decode_state.batch_info(kv_cache=cache, prefill=True)
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
        binfo = state.decode_state.batch_info(kv_cache=state.cache)
        logits, cache = model.decode(binfo.tokens, state.cache, binfo, binfo.pos_ids)

        seed = jax.random.PRNGKey(state.step)
        seed = jnp.tile(seed, binfo.max_seqs)
        seed = seed.reshape(binfo.max_seqs, 2)
        prng_keys = jax.vmap(jax.random.fold_in)(seed, binfo.new_token_dests.array)
        temps = state.decode_state.temperature
        new_tokens, logprobs = hax.vmap(sampler, "position")(logits, temps, key=prng_keys)
        
        # Update decode state with the sampled tokens
        decode_state = state.decode_state.update_tokens(new_tokens=new_tokens, new_logprobs=logprobs, step=state.step)
        outputs = state.outputs.update(new_tokens=new_tokens, new_logprobs=logprobs, step=state.step)

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
        logprobs=jnp.zeros((decode_state.max_seqs, num_rounds), dtype=jnp.float32),
        tokens=jnp.zeros((decode_state.max_seqs, num_rounds), dtype=jnp.int32),
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
            config = dataclasses.replace(
                config, max_pages=(config.max_seqs * max(1, config.max_seq_len // config.page_size))
            )
            assert config.max_pages > 0, "Imputed max_pages must be > 0"  # type: ignore

        spec = PageTableSpec(max_seqs=config.max_seqs, page_size=config.page_size, num_pages=config.max_pages)  # type: ignore
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

    def generate(self, requests: Sequence[Request]) -> GenerationResult:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).

        Args:
            requests: Sequence of generation requests
        """
        # Validate batch size doesn't exceed prefill capacity
        if len(requests) > self.config.max_seqs:
            raise ValueError(
                f"Batch size ({len(requests)}) exceeds max_seqs ({self.config.max_seqs}). "
                "Decompose your request into smaller batches or increase max_seqs."
            )

        # validate we don't have any sequences with n_generations exceeding max_seqs
        max_needed = max(int(r.n_generations) for r in requests)
        if max_needed > int(self.page_spec.max_seqs):
            raise ValueError(
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.page_spec.max_seqs})."
                "Decompose your request into smaller batches or increase max_seqs when building the service."
            )

        # Build prefill state with static shapes and run prefill to populate KV cache
        decode_state = build_prefill_state(
            requests,
            self.page_spec,
            token_buckets=self.config.prefill_token_buckets,
        )
        self.cache = _run_prefill(self.cache, decode_state, self.model)

        # Build decode state for autoregressive generation, handling clones
        prefill_token_dests = jax.device_get(decode_state.token_dests).array
        decode_state = build_decode_state(requests, self.page_spec, prefill_token_dests)
        num_seqs = decode_state.num_seqs

        # Outer generation loop: run until all sequences finish or we hit max length
        min_seq_len = min(len(req.prompt_tokens) for req in requests)
        max_outer_rounds = max(1, (self.config.max_seq_len - min_seq_len) // self.config.tokens_per_round)
        all_tokens = []
        all_logprobs = []
        final_position = np.full(num_seqs, -1, dtype=np.int32)

        for outer_round in range(max_outer_rounds):
            logger.info(f"[OUTER_LOOP] Starting outer_round={outer_round} of {max_outer_rounds}")
            self.cache, decoded_outputs, decode_state = _run_generation_loop(
                page_cache=self.cache,
                decode_state=decode_state,
                sampler=self.sampler,
                model=self.model,
                num_rounds=self.config.tokens_per_round,
            )

            outputs = jax.device_get(decoded_outputs)
            all_tokens.append(outputs.tokens)
            all_logprobs.append(outputs.logprobs)

            # Concatenate all tokens generated so far for each sequence
            tokens_so_far = np.concatenate(all_tokens, axis=1)
            final_position = _check_stop_sequences(tokens_so_far, requests, final_position)

            # Check if all sequences are done
            if np.all(final_position > -1):
                logger.info(f"All sequences finished after {outer_round + 1} outer rounds")
                break

        # Aggregate all tokens across rounds
        all_tokens_concat = np.concatenate(all_tokens, axis=1)  # [num_seqs, total_rounds]
        all_logprobs_concat = np.concatenate(all_logprobs, axis=1)

        total_generated = 0

        seq_offset = 0
        tokens_list = []
        logprobs_list = []
        for req in requests:
            for _ in range(req.n_generations):
                seq_tokens = all_tokens_concat[seq_offset]
                seq_logprobs = all_logprobs_concat[seq_offset]
                # slice out anything beyond the stop position. our kv-cache writes
                # roll over our input tokens if we proceed past max_seq_len, so always
                # cap there.
                if final_position[seq_offset] == -1:
                    valid_len = self.config.max_seq_len - len(req.prompt_tokens)
                else:
                    valid_len = final_position[seq_offset]

                tokens_list.append(seq_tokens[:valid_len].tolist())
                logprobs_list.append(seq_logprobs[:valid_len].tolist())
                total_generated += valid_len
                seq_offset += 1

        return GenerationResult(
            tokens=tokens_list,
            logprobs=logprobs_list,
            total_generated=total_generated,
        )
