# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
This module contains code for running the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
inside Levanter runs. The EleutherAI LM Evaluation Harness is a tool for evaluating language models on a variety of tasks.

The [run_lm_eval_harness][] function runs the EleutherAI LM Evaluation Harness on a given model and tasks, and returns the
results.

It can also be used as a callback, via the [lm_eval_harness][] function.

Note that Levanter does not support generation (use VLLM or something) and the [generate_until][] method is not implemented.
So we only support tasks that work with loglikelihood, which is most(?) of them.

References:

* https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
* https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/TPU_cluster.py#L6

"""

import dataclasses
import json
import logging
import tempfile
import pathlib
import typing
from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from jax.sharding import PartitionSpec

import haliax
from haliax import NamedArray

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.data.packing import (
    PromptCompletion,
    greedy_pack_prompt_completions,
    per_segment_correct,
    per_segment_loss,
)
from levanter.models.gpt2 import Gpt2Config
from levanter.models.loss import next_token_loss
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.hf_utils import HfTokenizer
from levanter.utils.py_utils import set_global_rng_seeds
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request as GenRequest
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.utils import INVALID

try:
    from lm_eval import evaluator
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import TemplateLM
    from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text
except ImportError:
    TemplateLM = object
    Instance = object
    evaluator = object
    handle_stop_sequences = None
    postprocess_generated_text = None

from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping, round_axis_for_partitioning

import levanter.config
from levanter.callbacks import StepInfo
from levanter.checkpoint import load_checkpoint
from levanter.data import batched
from levanter.data.loader import stack_batches
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import broadcast_shard, use_cpu_device
from levanter.utils.tree_utils import inference_mode
from levanter.books.util import (
    create_pz_histogram,
    create_pz_histogram_linear,
)

# Custom eval task support (Pz)
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase


logger = logging.getLogger(__name__)


# OK, so LM-Eval-Harness is not deterministic. This means we can't just run it on different workers and expect the
# order of requests to be the same. Sorting doesn't even seem to be correct (?!?!?) so we need to only run it on one
# process.
# This is our design:
# 1. Process 0 creates an LevanterHarnessLM object.
# 2. On all processes, we start a loop that waits for a request using jnp_broadcast_one_to_all
# 3. When a request is received (and it's not STOP) we process the request. The results are broadcast to all
#    devices, and process 0 records htem.
# 4. When a STOP request is received, we stop the loop and process 0 returns the results.


class _LmEvalHarnessWorker:
    """
    Worker for running the LM Eval Harness. Each worker process will run a copy of this class.
    The head process will run the main harness and dispatch requests to the workers while the
    others run in a loop waiting for requests.
    """

    def __init__(
        self,
        EvalBatch,
        EvalPos,
        model,
        axis_resources,
        tokenizer,
        mp,
        max_packed_segments,
        generation_kwargs=None,
    ):
        self.tokenizer = tokenizer
        self.max_packed_segments = max_packed_segments
        self.EvalBatch = EvalBatch
        self.EvalPos = EvalPos
        self.model = model
        self.axis_resources = axis_resources
        self.mp = mp
        self.max_packed_segments = max_packed_segments
        self._generation_kwargs = generation_kwargs or {"max_gen_toks": 256, "temperature": 0.0, "n": 1, "seed": None}

        self._dummy_batch = _make_dummy_batch(EvalBatch, EvalPos)

        def _eval_loglikelihood(
            model: LmHeadModel, packed_example: LmExample
        ) -> tuple[NamedArray, NamedArray, NamedArray]:
            """
            Returns:
                - segments: The segment IDs of the completions. (shape: (Segments,))
                - loss: The log-likelihood of the completion. (shape: (Segments,))
                - correct: Whether the completion is correct or not. (shape: (Segments,))
            """

            if self.mp is not None:
                model = self.mp.cast_to_compute(model)

            logits = model(packed_example.tokens, attn_mask=packed_example.attn_mask)
            logits = logits.astype(jnp.float32)
            Pos = logits.resolve_axis(self.EvalPos.name)

            loss = next_token_loss(
                Pos=Pos,
                Vocab=model.Vocab,
                logits=logits,
                true_ids=packed_example.tokens,
                loss_mask=packed_example.loss_mask,
                reduction=None,
            )

            # We need to compute losses and also whether or not the completion is correct
            # (i.e. the greedy prediction is the target)
            pred_targets = hax.argmax(logits, axis=model.Vocab)
            targets = hax.roll(packed_example.tokens, -1, axis=Pos)
            is_correct = targets == pred_targets

            # we need + 1 because we use -1 as a padding value for segments
            max_Segments = hax.Axis("Segments", size=self.max_packed_segments + 1)

            batched_segment_ids, batched_per_segment_losses = hax.vmap(per_segment_loss, self.EvalBatch)(
                packed_example, loss, max_Segments
            )

            _, batched_per_segment_correct = hax.vmap(per_segment_correct, self.EvalBatch)(
                packed_example, is_correct, max_Segments
            )

            segments = hax.flatten(batched_segment_ids, "segment")
            losses = hax.flatten(batched_per_segment_losses, "segment")
            correct = hax.flatten(batched_per_segment_correct, "segment")

            return segments, -losses, correct

        # no sharded outputs
        self._jit_loglikelihood = hax.named_jit(
            _eval_loglikelihood, axis_resources=axis_resources, out_axis_resources={}
        )

    @property
    def max_gen_toks(self) -> int:
        """Backward compatibility property for max_gen_toks."""
        return self._generation_kwargs.get("max_gen_toks", 256)

    def make_harness_lm(self):
        if jax.process_index() == 0:
            return LevanterHarnessLM(self)
        else:
            raise ValueError("Only process 0 can create the harness")

    def worker_message_loop(self):
        while True:
            message = self._receive_message()

            if message == _Message.STOP:
                return
            elif message == _Message.LOGLIKELIHOOD:
                payload = self._receive_payload()
                self.process_loglikelihood(payload)
            else:
                raise ValueError(f"Unknown message type: {message}")

    def _receive_message(self):
        stop_message = jnp.array(_Message.STOP)
        message = broadcast_shard(stop_message, PartitionSpec())
        return message.item()

    def _receive_payload(self):
        payload = broadcast_shard(
            self._dummy_batch,
            hax.partitioning.infer_resource_partitions(self._dummy_batch, preserve_existing_shardings=False),
        )
        return payload

    def _send_message(self, message):
        assert jax.process_index() == 0
        out = broadcast_shard(jnp.array(message), PartitionSpec())
        return out

    def _send_payload(self, payload):
        assert jax.process_index() == 0
        out = broadcast_shard(
            payload, hax.partitioning.infer_resource_partitions(payload, preserve_existing_shardings=False)
        )
        return out

    def process_loglikelihood(self, packed_request):
        out = self._jit_loglikelihood(self.model, packed_request)
        return out

    def dispatch_loglikelihood(self, packed_request):
        self._send_message(_Message.LOGLIKELIHOOD)
        self._send_payload(packed_request)
        return self.process_loglikelihood(packed_request)

    def stop(self):
        self._send_message(_Message.STOP)


class _Message:
    STOP = 0
    LOGLIKELIHOOD = 1


def _get_segments_this_batch(batch, max_segments_per_ex):
    unique_segs = np.unique(batch.attn_mask.segment_ids[0].array).tolist()
    # + 1 because we use -1 as a padding value for segments and allow that
    if len(unique_segs) > max_segments_per_ex + 1:
        raise ValueError(f"Too many segments in batch: {len(unique_segs)}")
    if -1 in unique_segs:
        unique_segs.remove(-1)

    return unique_segs


def _get_padding_count(batch, pad_token_id):
    # returns the total amount of padding in the batch
    padding_count = np.sum(batch.tokens.array == pad_token_id)
    total_tokens = batch.tokens.size
    return padding_count, total_tokens


class LevanterHarnessLM(TemplateLM):
    def __init__(self, leader: _LmEvalHarnessWorker):
        super().__init__()
        self.leader = leader
        # Storage for prompts and generations to include in outputs
        self.sample_outputs: dict[str, list[dict]] = {}

    tokenizer = property(lambda self: self.leader.tokenizer)
    EvalBatch = property(lambda self: self.leader.EvalBatch)
    EvalPos = property(lambda self: self.leader.EvalPos)

    @property
    def tokenizer_name(self) -> str:
        """Return a string identifier for the tokenizer/chat template."""
        if hasattr(self.tokenizer, "name_or_path"):
            return self.tokenizer.name_or_path
        elif hasattr(self.tokenizer, "model_name"):
            return self.tokenizer.model_name
        else:
            return "unknown_tokenizer"

    def chat_template(self, chat_template: str | None = None) -> str | None:
        """
        Return the chat template for this model.

        Args:
            chat_template: Optional override for chat template

        Returns:
            The chat template string, or None if not available
        """
        if chat_template is not None:
            return chat_template

        # Try to get the chat template from the tokenizer
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template

        # If no chat template is available, return None
        return None

    @property
    def max_gen_toks(self):
        return self.leader.max_gen_toks

    @property
    def generation_kwargs(self):
        """Get the generation kwargs from the worker."""
        return self.leader._generation_kwargs

    def apply_chat_template(self, chat_history: list[dict], **kwargs) -> str:
        """
        Apply chat template to format a conversation history.

        Args:
            chat_history: List of messages in the format [{"role": "user", "content": "..."}, ...]
            **kwargs: Additional arguments to pass to the tokenizer's apply_chat_template method

        Returns:
            Formatted string ready for tokenization
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
            **{k: v for k, v in kwargs.items() if k != "add_generation_prompt"},
        )

    @property
    def eot_token_id(self) -> int:
        """Return the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    def set_current_task(self, task_name: str):
        """Set the current task name for organizing sample outputs."""
        self._current_task = task_name
        if task_name not in self.sample_outputs:
            self.sample_outputs[task_name] = []

    def get_sample_outputs(self) -> dict[str, list[dict]]:
        """Get all stored sample outputs."""
        return self.sample_outputs

    def clear_sample_outputs(self):
        """Clear all stored sample outputs."""
        self.sample_outputs.clear()

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("_loglikelihood_tokens is not yet supported")

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        """
        if self.tokenizer.pad_token_id is None:
            logger.warning("No pad token set. Setting to eos token.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Store prompt-continuation pairs for output logging
        current_task = getattr(self, "_current_task", "loglikelihood_task")
        if current_task not in self.sample_outputs:
            self.sample_outputs[current_task] = []

        for request in requests:
            prompt = request.args[0]
            continuation = request.args[1]
            self.sample_outputs[current_task].append(
                {
                    "prompt": prompt,
                    "generation": continuation,  # For loglikelihood, the "generation" is the continuation being evaluated
                }
            )

        packed = _pack_requests(requests, self.tokenizer, self.EvalPos, self.leader.max_packed_segments)
        packed_iterator = stack_batches(iter(packed), self.EvalPos, self.EvalBatch)
        packed_iterator = BackgroundIterator(packed_iterator, max_capacity=1024)

        result_probs = np.zeros(len(requests))
        result_greedy = np.zeros(len(requests))
        covered_points = np.zeros(len(requests), dtype=bool)

        total_tokens_expected = len(packed) * self.EvalPos.size

        total_padding = 0
        total_tokens_seen = 0
        pbar = tqdm(total=total_tokens_expected, desc="loglikelihood", unit="tok")
        for q, batch in enumerate(packed_iterator):
            segments_this_batch = _get_segments_this_batch(
                batch, self.leader.max_packed_segments * self.EvalBatch.size
            )

            padding_count, batch_tokens = _get_padding_count(batch, self.tokenizer.pad_token_id)

            out_ids, out_lls, out_correct = self.leader.dispatch_loglikelihood(batch)

            out_ids = np.array(out_ids.array)
            out_lls = np.array(out_lls.array)
            out_correct = np.array(out_correct.array)
            # -1's are going to be where we had too few sequences to fill a batch
            valid_indices = out_ids != -1

            out_ids_this_batch = out_ids[valid_indices].tolist()

            missing_ids = set(segments_this_batch) - set(out_ids_this_batch)
            extra_ids = set(out_ids_this_batch) - set(segments_this_batch)
            assert len(missing_ids) == 0, f"Missing segments: {missing_ids}"
            assert len(extra_ids) == 0, f"Extra segments: {extra_ids}"

            result_probs[out_ids[valid_indices]] = out_lls[valid_indices]
            result_greedy[out_ids[valid_indices]] = out_correct[valid_indices]
            covered_points[out_ids[valid_indices]] = True

            total_padding += padding_count
            total_tokens_seen += batch_tokens

            pbar.set_postfix(
                padding=f"{total_padding}/{total_tokens_seen} = {(total_padding) / (total_tokens_seen):.2f}",
                this_padding=f"{padding_count}/{batch_tokens}= {padding_count / batch_tokens:.2f}",
            )
            pbar.update(batch_tokens)

        missing_points = np.where(~covered_points)[0]
        assert len(missing_points) == 0, f"Missing points: {missing_points}"

        result = list(zip(result_probs, result_greedy))
        logger.info(f"Finished running {len(requests)} loglikelihoods.")

        return result

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """
        Tokenize a string or list of strings.

        Args:
            string: The string(s) to tokenize.
            left_truncate_len: If provided, left-truncate the encoded tokens to this length.
            add_special_tokens: Whether to add special tokens during tokenization.
            truncation: Whether to enable tokenizer truncation.

        Returns:
            Token IDs as a list (for single string) or list of lists (for multiple strings).
        """
        if not add_special_tokens:
            add_special_tokens = False
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]  # type: ignore
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _process_and_tokenize_stop_sequences(self, until: Optional[List[str]], eos: str) -> Optional[NamedArray]:
        """
        Process and tokenize stop sequences for generation.

        Args:
            until: List of stop sequences, if any
            eos: End-of-sequence token string

        Returns:
            NamedArray with tokenized stop sequences, or None if no valid stop sequences
        """
        if not until:
            return None

        # Process stop sequences to ensure EOS is included
        processed_until = handle_stop_sequences(until, eos=eos)

        if not processed_until:
            return None

        # Tokenize all stop sequences
        all_stop_tokens = []
        for stop_seq in processed_until:
            stop_ids_list = self.tok_encode(stop_seq, add_special_tokens=False)
            if len(stop_ids_list) > 0:
                all_stop_tokens.append(stop_ids_list)

        if not all_stop_tokens:
            return None

        # Find the maximum length for padding
        max_len = max(len(tokens) for tokens in all_stop_tokens)

        # Left pad all sequences to the same length
        padded_tokens = []
        for tokens in all_stop_tokens:
            padding_needed = max_len - len(tokens)
            padded = [INVALID] * padding_needed + tokens
            padded_tokens.append(padded)

        # Convert to named array with proper dimensions
        stop_tokens_array = jnp.asarray(padded_tokens, dtype=jnp.int32)
        return haliax.named(stop_tokens_array, ("stop_seq", "position"))

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        # Error out on multihost JAX - Engine doesn't support it yet
        if jax.process_count() > 1:
            raise NotImplementedError(
                "InferenceEngine does not yet support multihost JAX. " "Please use a single host for generation tasks."
            )

        # print(f'len(requests)={len(requests)}')

        # Implement simple generation using InferenceEngine.
        # requests: list[Instance] where args[0] = prompt, args[1] may be stop strings (list[str])
        # kwargs may include max_gen_toks, temperature, n (n_generations), seed
        if self.tokenizer.pad_token_id is None:
            logger.warning("No pad token set. Setting to eos token.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Require a model with paged decode support
        if not hasattr(self.leader.model, "initial_cache") or not hasattr(self.leader.model, "decode"):
            raise NotImplementedError(
                "generate_until requires a model with paged decode support (initial_cache/decode)."
            )

        # Extract prompts and per-request params following vLLM pattern
        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        processed_kwargs_list: list[dict] = []

        for i, gen_kwargs in enumerate(all_gen_kwargs):
            # print(f'{gen_kwargs=}')

            # Copy and process generation kwargs
            processed_gen_kwargs = gen_kwargs.copy()

            # Apply defaults from generation_kwargs (user config) first
            for key, value in self.generation_kwargs.items():
                processed_gen_kwargs.setdefault(key, value)

            # Standardize kwargs using our _modify_gen_kwargs method
            processed_gen_kwargs = self._modify_gen_kwargs(processed_gen_kwargs)
            processed_kwargs_list.append(processed_gen_kwargs)

        # Tokenize prompts and compute capacity needs
        prompt_token_lists: list[list[int]] = self.tok_encode(context, add_special_tokens=False)  # type: ignore

        # Truncate from left if needed to fit model max length, accounting for generation tokens
        max_length = self.EvalPos.size
        for i, (toks, gen_kwargs) in enumerate(zip(prompt_token_lists, processed_kwargs_list)):
            # Reserve space for generation tokens
            max_gen_toks = gen_kwargs["max_gen_toks"]
            max_ctx_len = max_length - max_gen_toks

            if len(toks) > max_ctx_len:
                overflow = len(toks) - max_ctx_len
                logger.warning(f"Prompt {i} too long ({len(toks)}). Truncating left by {overflow}.")
                prompt_token_lists[i] = toks[-max_ctx_len:]

        # Process stop sequences for each request individually
        # Get EOS token for stop sequence handling
        eos = self.tokenizer.decode(self.eot_token_id)

        # Process stop sequences and tokenize them for each request
        for gen_kwargs in processed_kwargs_list:
            gen_kwargs["stop_tokens"] = self._process_and_tokenize_stop_sequences(gen_kwargs.get("until"), eos)

        # Calculate max stop sequences and tokens from all requests
        max_stop_seqs = 4
        max_stop_tokens = 16

        for gen_kwargs in processed_kwargs_list:
            stop_tokens = gen_kwargs.get("stop_tokens")
            if stop_tokens is not None:
                # stop_tokens is a NamedArray with shape (stop_seq, position)
                num_stop_seqs = stop_tokens.shape["stop_seq"]
                num_stop_tokens = stop_tokens.shape["position"]
                max_stop_seqs = max(max_stop_seqs, num_stop_seqs)
                max_stop_tokens = max(max_stop_tokens, num_stop_tokens)

        # Taken from: config/sampler/sample_llama8b.yaml
        engine_cfg = InferenceEngineConfig(
            max_stop_seqs=max_stop_seqs,
            max_stop_tokens=max_stop_tokens,
            max_pages=16384,
            max_seqs=256,
            page_size=8,
            max_pages_per_seq=512,
            compute_dtype=jnp.bfloat16,
            max_queued_tokens=256,
            max_seqs_in_prefill=16,
            max_prefill_size=max_length,
        )

        engine = InferenceEngine.from_model_with_config(
            model=self.leader.model, tokenizer=self.tokenizer, config=engine_cfg
        )

        # Build generation requests
        base_key = jrandom.PRNGKey(engine_cfg.seed)
        gen_requests: list[GenRequest] = []
        for i, (toks, gen_kwargs) in enumerate(zip(prompt_token_lists, processed_kwargs_list)):
            # Extract parameters from processed kwargs
            max_gen_toks = gen_kwargs["max_gen_toks"]
            temperature = gen_kwargs["temperature"]
            n_generations = gen_kwargs["n"]
            seed = gen_kwargs.get("seed")
            stop_tokens = gen_kwargs.get("stop_tokens")
            # print(f'{temperature=}')
            # print(f'{stop_tokens=}')

            # Create sequence decoding parameters
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(toks) + max_gen_toks, dtype=jnp.int32),
                stop_tokens=stop_tokens,
                temperature=jnp.array(temperature, dtype=jnp.float32),
                key=jrandom.fold_in(base_key if seed is None else jrandom.PRNGKey(seed), i),
            )
            gen_requests.append(
                GenRequest(
                    prompt_tokens=list(map(int, toks)),
                    request_id=i,
                    decode_params=seq_params,
                    n_generations=int(n_generations),
                    enable_logprobs=False,
                )
            )

        result = engine.generate(gen_requests)

        # Decode first generation per request (LM Harness expects one string per request)
        outputs: list[str] = []
        output_idx = 0
        for i, (toks, gen_kwargs) in enumerate(zip(prompt_token_lists, processed_kwargs_list)):
            # Consume one sequence output per request
            if output_idx < len(result.tokens):
                full_tokens = result.tokens[output_idx]
                # Engine tokens are generated tokens only (prompt not included)
                text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)

                # Post-process the generated text using the imported utility function
                text = postprocess_generated_text(
                    text, gen_kwargs.get("until"), None  # think_end_token - could be made configurable if needed
                )
                outputs.append(text)
                output_idx += 1  # consume one generation per request
            else:
                text = ""
                outputs.append(text)

            # Store prompt and generation for output logging
            current_task = getattr(self, "_current_task", "generation_task")
            if current_task not in self.sample_outputs:
                self.sample_outputs[current_task] = []
            # Decode the prompt for storage
            prompt_text = self.tokenizer.decode(toks, skip_special_tokens=False)
            self.sample_outputs[current_task].append(
                {
                    "prompt": prompt_text,
                    "generation": text,
                }
            )

        # print(f'{outputs=}')

        return outputs

    @staticmethod
    def _modify_gen_kwargs(kwargs: dict) -> dict:
        """
        Modify generation kwargs to standardize parameters, similar to vLLM implementation.
        """
        # Handle temperature
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            kwargs["temperature"] = max(0.0, float(kwargs["temperature"]))
        else:
            kwargs.setdefault("temperature", 0.0)

        # Handle do_sample parameter like vLLM does
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False:
            if kwargs["temperature"] == 0.0:
                logger.debug("Got `do_sample=False` with temperature 0.0, ensuring deterministic sampling")
            elif kwargs["temperature"] > 0.0:
                raise ValueError(
                    f"Conflicting parameters: do_sample=False but temperature={kwargs['temperature']} > 0.0. "
                    f"For deterministic sampling, set temperature=0.0 or remove do_sample=False."
                )

        # Handle max_gen_toks parameter
        if "max_gen_toks" in kwargs and kwargs["max_gen_toks"] is not None:
            kwargs["max_gen_toks"] = int(kwargs["max_gen_toks"])
        else:
            kwargs.setdefault("max_gen_toks", 256)

        # Handle n generations parameter
        if "n" in kwargs and kwargs["n"] is not None:
            kwargs["n"] = int(kwargs["n"])
        else:
            kwargs.setdefault("n", 1)

        # Handle seed parameter
        if "seed" in kwargs and kwargs["seed"] is not None:
            kwargs["seed"] = int(kwargs["seed"])
        # Note: seed can remain None, which is valid

        return kwargs


@dataclass(frozen=True)
class TaskConfig:
    """
    This is a dataclass that represents the configuration for a task in the LM Eval Harness. It is used to specify
    the configuration for a task in the LM Eval Harness, and is used to generate the task dictionary that the LM Eval
    Harness expects.

    nb that LM Eval Harness has its own TaskConfig, but its defaults are not the same as just passing in
    a dict, and we want the behavior of passing in a dict.

    Nones are not included in the dictionary representation, and LM Eval Harness will use its own defaults for any
    missing values.

    Docs are copied from the LM Eval Harness task guide. The LM Eval Harness task guide is the authoritative source
    for what these fields do. They were copied as of 2024-12-03.

    See Also:
       * [LM Eval Harness TaskConfig](https://github.com/EleutherAI/lm-evaluation-harness/blob/0ef7548d7c3f01108e7c12900a5e5eb4b4a668f7/lm_eval/api/task.py#L55)
       * [LM Eval Harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#parameters)
    """

    task: str
    """ The name of the task to run."""
    task_alias: str | None = None
    """ An alias for the task. We log this name to wandb."""
    num_fewshot: int | None = None

    use_prompt: str | None = None
    """ Name of prompt in promptsource to use. if defined, will overwrite doc_to_text, doc_to_target, and doc_to_choice."""
    description: str | None = None
    """An optional prepended Jinja2 template or string which will be prepended to the few-shot examples passed into the model, often describing the task or providing instructions to a model, such as "The following are questions (with answers) about {{subject}}.\n\n". No delimiters or spacing are inserted between the description and the first few-shot example."""
    target_delimiter: str | None = None
    """String to insert between input and target output for the datapoint being tested. defaults to " " """
    fewshot_delimiter: str | None = None
    """ String to insert between few-shot examples. defaults to "\\n\\n" """
    doc_to_text: str | None = None
    """Jinja2 template string to process a sample into the appropriate input for the model."""
    doct_to_target: str | None = None
    """Jinja2 template string to process a sample into the appropriate target for the model."""
    doc_to_choice: str | None = None
    """Jinja2 template string to process a sample into a list of possible string choices for multiple_choice tasks. """

    def to_dict(self):
        base_dict = dataclasses.asdict(self)
        return {k: v for k, v in base_dict.items() if v is not None}


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    task_spec: list[TaskConfig | str]
    max_examples: int | None = None
    max_length: int | None = None
    log_samples: bool = False
    bootstrap_iters: int = 0
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    generation_kwargs: dict = dataclasses.field(
        default_factory=lambda: {"max_gen_toks": 256, "temperature": 0.0, "n": 1, "seed": None}
    )
    """
    Default generation parameters for text generation tasks.

    Supported parameters:
    - max_gen_toks: Maximum number of tokens to generate (default: 256)
    - temperature: Sampling temperature, 0.0 for deterministic (default: 0.0)
    - n: Number of completions to generate per prompt (default: 1)
    - seed: Random seed for generation, None for random (default: None)

    These can be overridden on a per-request basis by the evaluation harness.
    """

    @property
    def max_gen_toks(self) -> int:
        """Backward compatibility property for max_gen_toks."""
        return self.generation_kwargs.get("max_gen_toks", 256)

    def to_task_spec(self) -> list[str | dict]:
        return [task.to_dict() if isinstance(task, TaskConfig) else task for task in self.task_spec]

    def to_task_dict(self) -> dict:
        """
        Convert the task spec to a dictionary that the LM Eval Harness expects.

        This is a bit more complex than we'd like, because we want to run e.g. Hellaswag 0-shot and 10-shot in the same
        run, and LM Eval Harness doesn't seem to want to do that by default. So we need to do some hacky stuff to make
        it work.
        """
        logger.info("Loading tasks...")
        import lm_eval.tasks as tasks

        manager = tasks.TaskManager()
        # we need to do it this way b/c i can't figure out how to run e.g. hellaswag 0 shot and 10 shot in a single run
        this_tasks = {}
        for task in tqdm(self.to_task_spec()):
            try:
                # Skip custom eval_pz tasks here; handled separately
                if isinstance(task, dict) and task.get("task") == "eval_pz":
                    continue
                if isinstance(task, str):
                    this_tasks.update(tasks.get_task_dict(task, manager))
                else:
                    our_name = task.get("task_alias", task["task"]) if isinstance(task, dict) else task
                    our_name = our_name.replace(" ", "_")
                    tasks_for_this_task_spec = self._get_task_and_rename(manager, our_name, task)
                    for k, v in tasks_for_this_task_spec.items():
                        if k in this_tasks:
                            raise ValueError(f"Task {k} already exists")
                        this_tasks[k] = v
            except Exception as e:
                logger.exception(f"Failed to load task {task}")
                raise ValueError(f"Failed to load task {task}") from e

        logger.info(f"Loaded {len(this_tasks)} tasks")
        return this_tasks

    def _get_task_and_rename(self, manager, our_name, task: dict | str):
        """
        Get a task from the task manager and rename it to our_name.
        LM Eval Harness doesn't seem to want to run multiple instances of the same task with different fewshot settings,
        (or other differences) so we need to hack around that.
        """
        import lm_eval.tasks as tasks

        task_name = task if isinstance(task, str) else task["task"]

        task_dict = tasks.get_task_dict([task], manager)
        assert len(task_dict) == 1, f"Expected 1 task, got {len(task_dict)}"
        try:
            this_task = self._rename_tasks_for_eval_harness(task_dict, task_name, our_name)
        except AttributeError:
            logger.exception(f"Failed to rename task {task}: {task_dict}")
            raise ValueError(f"Failed to rename task {task}: {task_dict}")
        return this_task

    def _rename_tasks_for_eval_harness(self, this_task, lm_eval_task_name, our_name):
        import lm_eval.tasks as tasks

        # hacky, but this allows us to run multiple instances of the same task with different fewshot settings
        if isinstance(this_task, dict):
            out = {}
            for k, v in this_task.items():
                v = self._rename_tasks_for_eval_harness(v, lm_eval_task_name, our_name)

                if isinstance(k, tasks.ConfigurableGroup):
                    k._config.group = self._replace_name_with_our_name(k.group, lm_eval_task_name, our_name)
                    out[k] = v
                elif isinstance(k, str):
                    k = self._replace_name_with_our_name(k, lm_eval_task_name, our_name)
                    if isinstance(v, dict):
                        subtask_list = self._get_child_tasks(v)
                        # ok so inexplicably, lm_eval_harness doesn't wrap the key in a ConfigurableGroup when you pass
                        # in a task dict (it seems like a mistake), so we need to do that here
                        # subtask is the name of all of the child tasks in v
                        group = tasks.ConfigurableGroup(config={"group": k, "task": subtask_list})
                        out[group] = v
                    else:
                        out[k] = v
                else:
                    raise ValueError(f"Unknown key type: {k}")

            return out

        elif isinstance(this_task, tasks.ConfigurableTask):
            this_task.config.task = self._replace_name_with_our_name(
                this_task.config.task, lm_eval_task_name, our_name
            )
            return this_task
        else:
            raise ValueError(f"Unknown task type: {this_task}")

    def _replace_name_with_our_name(self, lm_eval_name, lm_eval_prefix, our_name_prefix):
        if our_name_prefix.startswith(lm_eval_prefix):
            suffix = our_name_prefix[len(lm_eval_prefix) :]
            prefix = lm_eval_prefix
        else:
            suffix = ""
            prefix = our_name_prefix
        if lm_eval_prefix in lm_eval_name:
            lm_eval_name = lm_eval_name.replace(lm_eval_prefix, prefix) + suffix
        else:
            lm_eval_name = prefix + "_" + lm_eval_name + suffix
        return lm_eval_name

    def _get_child_tasks(self, task_group):
        import lm_eval.tasks as tasks

        out = []
        for k, v in task_group.items():
            if isinstance(k, tasks.ConfigurableGroup):
                subtask_or_tasks = k.config.task
                if isinstance(subtask_or_tasks, str):
                    out.append(subtask_or_tasks)
                else:
                    out.extend(subtask_or_tasks)
            elif isinstance(k, str):
                out.append(k)
            else:
                raise ValueError(f"Unknown key type: {k}")

        return out


@dataclass(frozen=True)
class EvalHarnessMainConfig:
    eval_harness: LmEvalHarnessConfig
    tokenizer: str
    checkpoint_path: str
    checkpoint_is_hf: bool = False
    """If True, the checkpoint is a HuggingFace checkpoint. Otherwise, it is a Levanter checkpoint."""
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    """
    Whether or not to apply the chat template this model was trained with before running inference
    """
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    model: LmConfig = dataclasses.field(default_factory=Gpt2Config)

    @property
    def EvalBatch(self):
        return self.trainer.EvalBatch

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


def run_lm_eval_harness(
    config: LmEvalHarnessConfig,
    model,
    tokenizer,
    EvalBatch,
    axis_resources,
    mp: jmp.Policy | None,
) -> dict | None:
    """
    Run the LM Eval Harness on the given model and tasks.

    Returns:
        If running on process 0, returns the outputs of the LM Eval Harness with the following extra keys.
        - "averages": A dictionary with macro and micro averages for all metrics.
        Otherwise, returns None.
    """
    tasks_to_run = config.to_task_dict()
    # Extract custom eval_pz specs (if any)
    pz_specs = []
    for spec in config.to_task_spec():
        if isinstance(spec, dict) and spec.get("task") == "eval_pz":
            pz_specs.append(spec)

    outputs = _actually_run_eval_harness(
        config,
        model,
        tasks_to_run,
        tokenizer,
        EvalBatch,
        axis_resources,
        mp,
        pz_specs=pz_specs,
        data_config=None,
        decode_once_state=None,
    )

    return outputs


def _actually_run_eval_harness(
    config: LmEvalHarnessConfig,
    model: LmHeadModel,
    tasks_to_run: dict,
    tokenizer: HfTokenizer,
    EvalBatch: haliax.Axis,
    axis_resources: ResourceMapping,
    mp: jmp.Policy | None,
    *,
    pz_specs: Optional[list[dict]] = None,
    data_config: Optional[Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase]] = None,
    decode_once_state: Optional[dict] = None,
) -> dict | None:
    """
    Actually run the LM Eval Harness on the given model and tasks. This is a separate function so that it can be used
    by the main function and the callback function.

    Returns:
        The outputs of the LM Eval Harness with the following extra keys:
        - "averages": A dictionary with macro and micro averages for all metrics.

    """
    max_examples = config.max_examples
    max_length = config.max_length

    EvalPos = model.Pos if max_length is None else model.Pos.resize(max_length)
    num_parameters = levanter.utils.jax_utils.parameter_count(model)
    logger.info(
        f"Evaluating with max length {EvalPos.size} and batch size {EvalBatch.size}. There are"
        f" {num_parameters} parameters in the model."
    )
    logger.info("Running eval harness...")

    worker = _LmEvalHarnessWorker(
        EvalBatch,
        EvalPos,
        model,
        axis_resources,
        tokenizer,
        mp,
        max_packed_segments=64,
        generation_kwargs=config.generation_kwargs,
    )

    if jax.process_index() == 0:
        logger.info("Process 0 is running the eval harness.")
        harness = worker.make_harness_lm()

        # eval_harness only sets seeds in simple_evaluate, which we can't use (I think?)
        tasks_to_run = _adjust_config(tasks_to_run, 0)

        # Clear any previous sample outputs
        harness.clear_sample_outputs()

        outputs: dict = {"results": {}, "n-samples": {}}

        if len(tasks_to_run) > 0:
            with set_global_rng_seeds(0):
                lm_outputs = evaluator.evaluate(
                    harness,
                    tasks_to_run,
                    limit=max_examples,
                    log_samples=config.log_samples,
                    bootstrap_iters=config.bootstrap_iters,
                    apply_chat_template=config.apply_chat_template,
                    fewshot_as_multiturn=config.fewshot_as_multiturn,
                )
            worker.stop()

            if lm_outputs is not None:
                # Merge core outputs except results/n-samples which we merge explicitly
                for k, v in lm_outputs.items():
                    if k in ("results", "n-samples"):
                        continue
                    outputs[k] = v
                outputs["results"].update(lm_outputs.get("results", {}))
                outputs["n-samples"].update(lm_outputs.get("n-samples", {}))

        # Run custom eval_pz tasks if requested
        if pz_specs:
            pz_results, pz_counts = _run_eval_pz_tasks(
                model=model,
                tokenizer=tokenizer,
                axis_resources=axis_resources,
                mp=mp,
                data_config=data_config,
                specs=pz_specs,
                decode_once_state=decode_once_state,
            )
            outputs["results"].update(pz_results)
            outputs["n-samples"].update(pz_counts)

        # Calculate averages for LM-Eval tasks only (skip eval_pz)
        if len(outputs.get("results", {})) > 0:
            averages = _compute_averages(outputs)
            outputs["averages"] = averages

        return outputs
    else:
        logger.info(f"Process {jax.process_index()} is waiting for eval harness requests from process 0.")
        worker.worker_message_loop()

        logger.info("Finished running eval harness.")

        return None


def _run_eval_pz_tasks(
    *,
    model: LmHeadModel,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp: jmp.Policy | None,
    data_config: Optional[Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase]],
    specs: list[dict],
    decode_once_state: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Execute custom eval_pz specs and return (results, n-samples) dicts to merge into LM-Eval outputs."""
    results: dict = {}
    n_samples: dict = {}

    if data_config is None:
        logger.warning("eval_pz tasks specified but no data_config provided; skipping.")
        return results, n_samples

    # Build caches and access the input_ids store
    try:
        caches = data_config.build_caches("train", monitors=False)
    except Exception:
        logger.exception("Failed to build caches for eval_pz; skipping eval_pz tasks.")
        return results, n_samples

    cmapping = axis_resources
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _compute_logprob_for_tokens(tokens_1d: np.ndarray, prompt_len: int) -> float:
        N = int(tokens_1d.shape[0])
        Pos = model.Pos.resize(N)
        toks_named = haliax.named(np.array(tokens_1d, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, toks_named, prompt_length=int(prompt_len), ignore_id=pad_id)

        m = model
        if mp is not None:
            m = mp.cast_to_compute(m)

        with haliax.axis_mapping(cmapping):
            logits = m(ex.tokens, attn_mask=ex.attn_mask)
            logits = logits.astype(jnp.float32)
            nll = next_token_loss(
                Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None
            )
            total_nll = haliax.sum(nll, axis=Pos).array
        return -float(np.array(total_nll))

    for spec in specs:
        if not (isinstance(spec, dict) and spec.get("task") == "eval_pz"):
            continue
        alias = spec.get("task_alias")
        datasets = spec.get("datasets")
        doc_tokens = spec.get("doc_tokens")  # None or int
        chunk_size = int(spec.get("chunk_size", 0))
        if chunk_size <= 0:
            logger.warning(f"eval_pz missing/invalid chunk_size: {spec}")
            continue
        prompt_tokens = int(spec.get("prompt_tokens", chunk_size // 2))
        suffix_tokens = chunk_size - prompt_tokens
        cursor_inc_tokens = int(spec.get("cursor_inc_tokens", 1))
        histogram = bool(spec.get("histogram", False))
        histogram_linear = bool(spec.get("histogram_linear", True))
        pz_threshold = float(spec.get("pz_threshold", 1e-4))
        pz_npz = bool(spec.get("pz_npz", False))
        decode_preview = spec.get("decode_preview")
        verify_treecache = bool(spec.get("verify_treecache", False))

        # Select dataset names
        if datasets is None:
            selected = list(caches.keys())
        else:
            selected = [name for name in datasets if name in caches]
            missing = [name for name in datasets if name not in caches]
            for m in missing:
                logger.warning(f"eval_pz requested dataset '{m}' not found in caches; skipping.")
        if not selected:
            continue

        for ds_name in selected:
            task_key = alias or f"eval_pz/{ds_name}"
            if alias and len(selected) > 1:
                task_key = f"eval_pz/{alias}/{ds_name}"

            cache = caches[ds_name]
            try:
                input_store = cache.store.tree["input_ids"]  # type: ignore[index]
            except Exception:
                logger.warning(f"Dataset '{ds_name}' cache missing input_ids; skipping.")
                continue
            try:
                first_ids = np.asarray(input_store[0], dtype=np.int32).reshape(-1)
            except Exception as e:
                logger.warning(f"Failed reading first document for '{ds_name}': {e}")
                continue

            # Determine document slice length
            if doc_tokens is None:
                eval_len = int(first_ids.shape[0])
            else:
                eval_len = int(min(int(doc_tokens), int(first_ids.shape[0])))
            doc_slice = first_ids[:eval_len]

            # Build sliding windows across the document slice
            pz_values: list[float] = []
            span_ranges: list[tuple[int, int]] = []
            if eval_len == 0:
                logger.warning(f"Dataset '{ds_name}' first document is empty; skipping eval_pz.")
                continue

            starts = list(range(0, max(eval_len - chunk_size, 0) + 1, max(1, cursor_inc_tokens)))
            if not starts:
                starts = [0]

            for s in starts:
                window = doc_slice[s : s + chunk_size]
                if window.shape[0] < chunk_size:
                    pad_len = chunk_size - window.shape[0]
                    window = np.concatenate([window, np.full((pad_len,), pad_id, dtype=np.int32)], axis=0)
                logprob = _compute_logprob_for_tokens(window, prompt_tokens)
                try:
                    prob = float(np.exp(logprob))
                except OverflowError:
                    prob = 0.0
                pz_values.append(prob)
                span_ranges.append((s, min(s + chunk_size - 1, eval_len - 1)))

            # Aggregate metrics
            if len(pz_values) == 0:
                mean_pz = median_pz = max_pz = 0.0
            else:
                arr = np.asarray(pz_values, dtype=np.float64)
                mean_pz = float(np.mean(arr))
                median_pz = float(np.median(arr))
                max_pz = float(np.max(arr))

            results[task_key] = {
                "num_windows": int(len(pz_values)),
                "mean_pz": mean_pz,
                "median_pz": median_pz,
                "max_pz": max_pz,
                "chunk_size": int(chunk_size),
                "prompt_tokens": int(prompt_tokens),
                "suffix_tokens": int(suffix_tokens),
                "cursor_inc_tokens": int(cursor_inc_tokens),
                "doc_len": int(eval_len),
            }
            n_samples[task_key] = len(pz_values)

            # Histogram artifact (optional)
            if histogram and len(pz_values) > 0 and jax.process_index() == 0:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_hist_path = tmp_file.name
                title = f"{ds_name} - P(z) over first doc"
                if histogram_linear:
                    _ = create_pz_histogram_linear(
                        pz_list=np.asarray(pz_values),
                        threshold=pz_threshold,
                        save_path=temp_hist_path,
                        book_title=title,
                    )
                else:
                    _ = create_pz_histogram(
                        pz_list=np.asarray(pz_values),
                        threshold=pz_threshold,
                        save_path=temp_hist_path,
                        book_title=title,
                    )
                # log artifact
                levanter.tracker.current_tracker().log_artifact(
                    temp_hist_path, name=f"pz_hist_{ds_name}.png", type="plot"
                )
                pathlib.Path(temp_hist_path).unlink(missing_ok=True)

            # NPZ artifact (optional)
            if pz_npz and len(pz_values) > 0 and jax.process_index() == 0:
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_npz:
                    np.savez(
                        tmp_npz.name,
                        pz_values=np.asarray(pz_values, dtype=np.float32),
                        span_ranges=np.asarray(span_ranges, dtype=np.int32),
                        config_info=np.asarray(
                            [chunk_size, prompt_tokens, cursor_inc_tokens, eval_len], dtype=np.int32
                        ),
                    )
                    tmp_npz_path = tmp_npz.name
                levanter.tracker.current_tracker().log_artifact(
                    tmp_npz_path, name=f"pz_values_{ds_name}.npz", type="data"
                )
                pathlib.Path(tmp_npz_path).unlink(missing_ok=True)

            # Decode preview once
            if decode_preview and decode_once_state is not None:
                state_key = f"{task_key}__decoded"
                if not decode_once_state.get(state_key, False):
                    preview_len = int(min(int(decode_preview), eval_len))
                    preview_ids = doc_slice[:preview_len].tolist()
                    preview_text = tokenizer.decode(preview_ids, skip_special_tokens=False)
                    results[task_key]["preview_text"] = preview_text
                    results[task_key]["preview_token_sum"] = int(sum(preview_ids))
                    if verify_treecache:
                        results[task_key]["verify_treecache"] = True
                        results[task_key]["first_doc_len"] = int(first_ids.shape[0])
                    decode_once_state[state_key] = True

    return results, n_samples


def _compute_averages(outputs):
    """
    Compute macro and micro averages of all metrics.

    Args:
        outputs: Dictionary with results and samples:
                 - "results": Dictionary of task-level results.
                 - "n-samples" : Dictionary of task-level sample counts.



    Returns:
        Averages dictionary with macro and micro averages for all metrics.
    """
    averages = {}
    metric_keys = set()

    # Collect all possible metrics across tasks, skipping custom eval_pz entries
    for task_name, task_results in outputs["results"].items():
        if str(task_name).startswith("eval_pz"):
            continue
        metric_keys.update(k for k in task_results.keys() if "stderr" not in k and k != "alias")

    # Compute macro and micro averages
    for metric in metric_keys:
        # Collect valid tasks for this metric
        # We iterate over the n-samples because real tasks (as opposed to aggregates like "mmlu") have counts
        valid_tasks = [
            (outputs["results"][task_name].get(metric), outputs["n-samples"][task_name]["effective"])
            for task_name in outputs["n-samples"]
            if outputs["results"][task_name].get(metric, None) is not None
        ]

        if not valid_tasks:
            continue  # Skip metrics with no valid tasks

        # Separate metric values and weights
        metric_values, this_examples_per_task = zip(*valid_tasks)

        # Compute macro and micro averages
        averages["macro_avg_" + metric] = np.mean(metric_values)
        if sum(this_examples_per_task) > 0:
            averages["micro_avg_" + metric] = np.average(metric_values, weights=this_examples_per_task)

    return averages


def run_eval_harness_main(config: EvalHarnessMainConfig):
    config.trainer.initialize()
    tokenizer = config.the_tokenizer

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        model: LmHeadModel

        # initialize the model
        if config.checkpoint_is_hf:
            model_config = config.model
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.checkpoint_path, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type, ref=config.checkpoint_path, dtype=config.trainer.mp.compute_dtype  # type: ignore
            )
        else:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard(model, parameter_axis_mapping)

        model = typing.cast(LmHeadModel, inference_mode(model, True))

        logger.info("Running LM eval harness....")
        outputs = run_lm_eval_harness(
            config.eval_harness,
            model,
            tokenizer,
            config.EvalBatch,
            axis_resources=compute_axis_mapping,
            mp=config.trainer.mp,
        )

        logger.info("Finished running LM eval harness")

        # log the results
        if jax.process_index() == 0:
            logger.info("Logging results to tracker")
            assert outputs is not None
            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())
            logger.info("Finished logging results to tracker")

            # log the results as json
            logger.info("uploading artifacts...")
            with open("lm_eval_harness_results.json", "w") as f:
                json.dump(outputs, f, indent=2)
                f.flush()
                f_path = f.name
                levanter.tracker.current_tracker().log_artifact(f_path, name="lm_eval_harness_results")

            print(json.dumps(outputs, indent=2), flush=True)

        return outputs


def log_report_to_tracker(prefix: str, report: dict, tracker: Optional[levanter.tracker.Tracker] = None):
    if tracker is None:
        tracker = levanter.tracker.current_tracker()

    to_log = {}
    for task_name, task_results in report["results"].items():
        for metric_name, metric_value in task_results.items():
            # remove the ",none" suffix, which eval-harness adds by default for some reason
            if metric_name.endswith(",none"):
                metric_name = metric_name[:-5]

            if isinstance(metric_value, float | int):
                to_log[f"{prefix}/{task_name}/{metric_name}"] = metric_value

    if "averages" in report:
        for metric_name, metric_value in report["averages"].items():
            if isinstance(metric_value, float | int):
                if metric_name.endswith(",none"):
                    metric_name = metric_name[:-5]

                to_log[f"{prefix}/averages/{metric_name}"] = metric_value

    tracker.log(to_log, step=None)


def lm_eval_harness(
    config: LmEvalHarnessConfig,
    tokenizer,
    EvalBatch,
    axis_resources,
    mp: jmp.Policy | None,
    data_config: Optional[Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase]] = None,
):
    tasks_to_run = config.to_task_dict()
    pz_specs: list[dict] = [
        spec for spec in config.to_task_spec() if isinstance(spec, dict) and spec.get("task") == "eval_pz"
    ]

    # track decode-once behavior across invocations
    decode_once_state: dict = {}

    def lm_eval_harness(step: StepInfo, force=False):
        if step.step == 0 and not force:
            return

        model = step.eval_model
        logger.info("Running eval harness...")
        outputs = _actually_run_eval_harness(
            config,
            model,
            tasks_to_run,
            tokenizer,
            EvalBatch,
            axis_resources,
            mp,
            pz_specs=pz_specs,
            data_config=data_config,
            decode_once_state=decode_once_state,
        )
        logger.info("Finished running eval harness.")

        if jax.process_index() == 0:
            assert outputs is not None
            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())
            logger.info("Logged report to tracker")

            # don't delete b/c wandb will sometimes defer upload
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                import json

                json.dump(outputs, f)
                f.flush()
                levanter.tracker.current_tracker().log_artifact(
                    f.name, name=f"lm_eval_harness_results.{step.step}.json", type="lm_eval_output"
                )
                logger.info("Uploaded results to tracker")

    return lm_eval_harness


# lifted from lm-eval simple_evaluate
def _adjust_config(task_dict, fewshot_random_seed=0):
    adjusted_task_dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            adjusted_task_dict = {
                **adjusted_task_dict,
                **{task_name: _adjust_config(task_obj, fewshot_random_seed=fewshot_random_seed)},
            }

        else:
            # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
            task_obj.set_fewshot_seed(seed=fewshot_random_seed)

            adjusted_task_dict[task_name] = task_obj

    return adjusted_task_dict


def _iterate_tokenized_requests(
    requests: list[Instance], tokenizer: HfTokenizer, max_length: int, batch_size: int
) -> Iterator[PromptCompletion]:
    """
    Tokenize the requests and yield them as PromptCompletions, for packing into LmExamples.
    """
    contexts = [request.args[0] for request in requests]

    completions = [request.args[1] for request in requests]

    # Combine contexts and completions for full tokenization
    combined_texts = [context + completion for context, completion in zip(contexts, completions)]

    # Batch tokenization for combined and context separately
    for batch_indices in batched(range(len(requests)), batch_size):
        # Extract batch data
        combined_batch = [combined_texts[i] for i in batch_indices]
        context_batch = [contexts[i] for i in batch_indices]
        # Tokenize batched inputs
        combined_encodings = tokenizer(combined_batch, truncation=False, padding=False)
        context_encodings = tokenizer(context_batch, truncation=False, padding=False)

        for off in range(len(batch_indices)):
            i = batch_indices[off]
            context_enc = context_encodings["input_ids"][off]
            all_enc = combined_encodings["input_ids"][off]

            context_enc_len = len(context_enc)

            if len(all_enc) > max_length:
                logger.warning(f"Request {i} is too long. Truncating.")
                # Truncate from the left
                context_enc_len = len(context_enc) - (len(all_enc) - max_length)
                all_enc = all_enc[-max_length:]
                if context_enc_len < 0:
                    context_enc_len = 0
                    logger.warning("Prompt length is negative after truncation. Setting to 0.")
            yield PromptCompletion(ids=all_enc, prompt_length=context_enc_len, segment_id=i)


def _pack_requests(
    requests: list[Instance],
    tokenizer: HfTokenizer,
    Pos: hax.Axis,
    max_pack_size: int,
) -> list[LmExample]:
    packed_iterator = _iterate_tokenized_requests(requests, tokenizer, Pos.size, batch_size=128)
    # TODO: use a better packing algorithm?
    return greedy_pack_prompt_completions(
        Pos,
        packed_iterator,
        max_segments_per_example=max_pack_size,
        pad_token=tokenizer.pad_token_id,
    )


def _make_dummy_batch(EvalBatch, EvalPos):
    dummy_batch = hax.vmap(LmExample.causal, EvalBatch)(
        hax.zeros(EvalPos, dtype=jnp.int32),
        loss_mask=hax.zeros(EvalPos, dtype=jnp.int32),
        segment_ids=hax.zeros(EvalPos, dtype=jnp.int32),
    )
    out = hax.shard(dummy_batch, {})
    return out


if __name__ == "__main__":
    levanter.config.main(run_eval_harness_main)()
    print("Done", flush=True)
