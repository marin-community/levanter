# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import logging
import warnings
from collections import defaultdict
from typing import Callable, Mapping, Optional, Sequence, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import Mesh
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceMapping

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.data import AsyncDataset, DataLoader
from levanter.models.lm_model import LmExample, LmHeadModel, compute_next_token_loss
from levanter.utils.hf_utils import HfTokenizer, byte_length_of_token
from levanter.utils.logging import LoadingTimeTrackerIterator
from levanter.utils.stat_utils import Arrayish, RunningMean
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")
M = TypeVar("M")


@dataclasses.dataclass
class EvalResult:
    micro_avg_loss: float  # per token across all datasets
    macro_avg_loss: float  # average of per-dataset average losses
    tag_macro_losses: dict[str, float]  # per tag average-per-token loss
    tag_micro_losses: dict[str, float]  # per tag total loss, for "parent" tags
    total_eval_loading_time: float
    micro_bpb: Optional[float] = None
    macro_bpb: Optional[float] = None
    tag_macro_bpb: Optional[dict[str, float]] = None
    tag_micro_bpb: Optional[dict[str, float]] = None
    # New fields for per-token loss tracking
    per_token_loss: Optional[hax.NamedArray] = None  # Full array of average loss per vocab ID
    token_macro_loss: Optional[float] = None  # Macro average over *appeared* token losses


@dataclasses.dataclass
class _EvalRunningMeans:
    """Helper class to track running means for evaluation losses."""

    total_loss_running_mean: RunningMean
    token_count_running_mean: RunningMean
    bytes_per_token_running_mean: RunningMean
    tag_total_loss_running_mean: RunningMean
    tag_token_count_running_mean: RunningMean
    tag_bytes_per_token_running_mean: RunningMean
    # New field for per-token loss tracking
    per_token_loss_running_mean: RunningMean


# This class doesn't try to be async or work with incomplete datasets, because it's eval


class DomainTaggedDataset(AsyncDataset[tuple[T, hax.NamedArray]]):
    """Holds multiple datasets, each with its own domain tag. Also indexes the tags to enable easier aggregation."""

    tag_index: Mapping[str, int]

    @property
    def tags(self):
        return self.tag_to_index.keys()

    def __init__(
        self, datasets: Sequence[tuple[AsyncDataset[T], Sequence[str]]], max_examples_per_dataset: Optional[int] = None
    ):
        super().__init__()
        self.datasets = []
        self._max_examples_per_dataset = max_examples_per_dataset

        tag_index: dict[str, int] = {}
        for i, (dataset, tags) in enumerate(datasets):
            if not tags and len(datasets) > 1:
                warnings.warn("Dataset has no tags. Giving it an index")
                tags = [f"domain_{i}"]
            for tag in tags:
                if tag not in tag_index:
                    tag_index[tag] = len(tag_index)

            if self._max_examples_per_dataset:
                dataset = dataset.take(self._max_examples_per_dataset)

            self.datasets.append((dataset, tags))

        self.tag_to_index = tag_index
        self.Tag = hax.Axis("tag", len(self.tag_to_index))
        self._tag_arrays = self._compute_tag_arrays()
        self._offsets: Optional[np.ndarray] = None

    async def _get_offsets(self) -> np.ndarray:
        if self._offsets is None:
            lengths = await asyncio.gather(*[dataset.async_len() for dataset, _ in self.datasets])
            if self._max_examples_per_dataset is not None:
                lengths = [min(length, self._max_examples_per_dataset) for length in lengths]
            self._offsets = np.cumsum([0] + lengths)

        return self._offsets  # type: ignore

    def _compute_tag_arrays(self):
        tag_arrays = []
        for dataset, tags in self.datasets:
            indexed = [self.tag_to_index[tag] for tag in tags]
            tags = np.zeros(self.Tag.size, dtype=np.int32)
            tags[indexed] = 1
            tags = hax.named(tags, self.Tag)

            tag_arrays.append(tags)
        return tag_arrays

    async def async_len(self) -> int:
        return int((await self._get_offsets())[-1])

    async def getitem_async(self, item: int) -> tuple[T, hax.NamedArray]:
        offsets = await self._get_offsets()
        dataset_index = np.searchsorted(offsets, item, side="right") - 1
        offset = offsets[dataset_index]
        dataset, tags = self.datasets[dataset_index]
        return await dataset.getitem_async(int(item - offset)), self._tag_arrays[dataset_index]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[T, hax.NamedArray]]:
        # Chatgpt wrote this. pretty sure it's correct
        offsets = await self._get_offsets()
        original_order = np.argsort(indices)
        sorted_indices = np.array(indices)[original_order]
        dataset_indices = (np.searchsorted(offsets, sorted_indices, side="right") - 1).tolist()

        # Group indices by the dataset they belong to
        grouped_indices = defaultdict(list)
        for idx, dataset_index in zip(sorted_indices, dataset_indices):
            grouped_indices[dataset_index].append(int(idx - offsets[dataset_index]))

        # Retrieve the batch for each group
        batch_futures: list = []
        for dataset_index, dataset_indices in grouped_indices.items():
            dataset, tags = self.datasets[dataset_index]
            dataset_batch = dataset.get_batch(dataset_indices)
            batch_futures.append(dataset_batch)

        batch_groups = await asyncio.gather(*batch_futures)
        batch = []
        for dataset_index, dataset_batch in zip(grouped_indices.keys(), batch_groups):
            batch.extend([(item, self._tag_arrays[dataset_index]) for item in dataset_batch])

        # Reorder the batch to match the original order of indices
        batch = [batch[i] for i in np.argsort(original_order)]

        return batch

    async def final_length_is_known(self) -> bool:
        return all(await asyncio.gather(*[dataset.final_length_is_known() for dataset, _ in self.datasets]))

    def is_finite(self) -> bool:
        return all(dataset.is_finite() for dataset, _ in self.datasets)

    async def current_len(self) -> Optional[int]:
        # We currently require all datasets to be finished before we do anything with this dataset, so...
        return await self.async_len()


def cb_tagged_lm_evaluate(
    EvalBatch: hax.Axis,
    model: LmHeadModel,
    tagged_eval_sets: Sequence[tuple[AsyncDataset[LmExample], Sequence[str]]],
    tokenizer: Optional[HfTokenizer] = None,
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping = None,
    max_examples_per_dataset: Optional[int] = None,
    eval_current: bool = True,
    eval_ema: bool = True,
    prefix: str = "eval",
    mp: jmp.Policy = None,
) -> Callable[[StepInfo], None]:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting
    an overall score.

    Tags can be hierarchical, with "/" as a separator. We log both a micro and macro average loss
    for each tag.

    This function also tracks per-token loss (average loss for each individual token ID) and
    a 'token macro loss' (the average of these per-token losses for tokens that appeared).

    !!! note

        loss_fn should return *per-token* loss (shape [EvalBatch, Token])

    Args:
        EvalBatch: The axis for the evaluation batch (mostly for the batch size)
        model: The model to evaluate. A template model is passed to capture metadata like axes.
        tagged_eval_sets: A sequence of tuples, where each tuple is an `AsyncDataset` and a sequence of tags.
        tokenizer: The tokenizer to use for BPB calculation.
        device_mesh: The device mesh to use for evaluation.
        axis_mapping: The resource mapping to use for evaluation.
        max_examples_per_dataset: The maximum number of examples to use from each dataset.
        eval_current: Whether to evaluate the current model.
        eval_ema: Whether to evaluate the EMA model.
        prefix: The prefix to use for logging.
        mp: The mixed precision policy to use for evaluation.
    """

    if device_mesh is None:
        device_mesh = hax.partitioning.get_context().mesh

    if axis_mapping is None:
        axis_mapping = hax.partitioning.get_context().axis_mapping

    if mp is None:
        mp = jmp.get_policy("primitives")

    dataset = DomainTaggedDataset(tagged_eval_sets, max_examples_per_dataset=max_examples_per_dataset)
    Tag = dataset.Tag
    Vocab = model.Vocab

    def callback(info: StepInfo):
        if not eval_current and not eval_ema:
            return

        with jax.spmd_mode("allow_all"):
            if eval_current:
                logger.info(f"Evaluating current model at step {info.step}")
                result = evaluate_model(info.model, prefix, use_ema=False)
                log_eval_result(result)

            if eval_ema and info.ema_model is not None:
                logger.info(f"Evaluating EMA model at step {info.step}")
                result = evaluate_model(info.ema_model, prefix, use_ema=True)
                log_eval_result(result)

    @eqx.filter_jit
    def compute_and_accumulate_loss_step(model, running_means: _EvalRunningMeans, batch: LmExample):
        # compute loss and sharded device array
        losses = compute_next_token_loss(model, batch, key=None)

        # we don't need the tags on the device, but it's fine
        # loss mask is a bool array, but it's fine to multiply by it
        masked_loss = losses * batch.loss_mask
        # total loss for the batch
        total_loss = hax.sum(masked_loss)
        # total tokens for the batch that are not padding
        token_count = hax.sum(batch.loss_mask)

        new_total_loss_mean = running_means.total_loss_running_mean.update(total_loss, token_count)
        new_running_means = dataclasses.replace(running_means, total_loss_running_mean=new_total_loss_mean)

        new_token_count_mean = running_means.token_count_running_mean.update(token_count)
        new_running_means = dataclasses.replace(new_running_means, token_count_running_mean=new_token_count_mean)

        # total loss for each tag
        tag_loss = hax.sum(masked_loss.broadcast_axis(Tag) * batch.tags, batch.tokens.axes)
        tag_token_count = hax.sum(batch.loss_mask.broadcast_axis(Tag) * batch.tags, batch.tokens.axes)

        new_tag_total_loss_mean = running_means.tag_total_loss_running_mean.update(tag_loss, tag_token_count)
        new_running_means = dataclasses.replace(
            new_running_means, tag_total_loss_running_mean=new_tag_total_loss_mean
        )

        new_tag_token_count_mean = running_means.tag_token_count_running_mean.update(tag_token_count)
        new_running_means = dataclasses.replace(
            new_running_means, tag_token_count_running_mean=new_tag_token_count_mean
        )

        # bytes per token for the batch
        if tokenizer:
            Pos = batch.tokens.axes[-1]
            next_tokens = hax.roll(batch.tokens, -1, Pos)
            bytes_per_token = hax.vmap(byte_length_of_token, "tok_id")(next_tokens)
            bytes_per_token = hax.named(bytes_per_token, next_tokens.axes)
            masked_bytes_per_token = bytes_per_token * batch.loss_mask
            total_bytes = hax.sum(masked_bytes_per_token)

            new_bpt_mean = running_means.bytes_per_token_running_mean.update(total_bytes, token_count)
            new_running_means = dataclasses.replace(new_running_means, bytes_per_token_running_mean=new_bpt_mean)

            tag_bytes = hax.sum(masked_bytes_per_token.broadcast_axis(Tag) * batch.tags, batch.tokens.axes)
            new_tag_bpt_mean = running_means.tag_bytes_per_token_running_mean.update(tag_bytes, tag_token_count)
            new_running_means = dataclasses.replace(
                new_running_means, tag_bytes_per_token_running_mean=new_tag_bpt_mean
            )

        # per-token loss tracking
        Pos = batch.tokens.axes[-1]
        next_token_ids = hax.roll(batch.tokens, shift=-1, axis=Pos)

        one_hot_targets = hax.nn.one_hot(next_token_ids, Vocab, dtype=losses.dtype)
        masked_one_hot = one_hot_targets * batch.loss_mask
        # masked_losses is masked_loss reshaped to be compatible with masked_one_hot

        sum_axes = batch.tokens.axes

        masked_total_losses_per_vocab_id = hax.sum(
            masked_one_hot * masked_loss.broadcast_axis(Vocab), sum_axes
        )
        masked_token_counts_per_vocab_id = hax.sum(masked_one_hot, sum_axes)

        new_per_token_loss_mean = running_means.per_token_loss_running_mean.update(
            masked_total_losses_per_vocab_id, masked_token_counts_per_vocab_id
        )
        new_running_means = dataclasses.replace(
            new_running_means, per_token_loss_running_mean=new_per_token_loss_mean
        )

        return losses, new_running_means

    def evaluate_model(model_to_eval, prefix: str, use_ema: bool) -> EvalResult:
        model_to_eval = inference_mode(model_to_eval, True)
        model_to_eval = eqx.tree_inference(model_to_eval, True)

        # TODO: i think this is not ideal, should probably pass in the policy from the trainer
        # but i'm not sure how to do that easily.
        model_to_eval = mp.cast_to_compute(model_to_eval)

        data_loader = DataLoader(dataset, EvalBatch.size)

        pbar = tqdm(total=len(dataset), desc="Evaluating", leave=False)
        loading_time_tracker = LoadingTimeTrackerIterator(data_loader, pbar)

        running_means = _EvalRunningMeans(
            total_loss_running_mean=RunningMean(),
            token_count_running_mean=RunningMean(),
            bytes_per_token_running_mean=RunningMean(),
            tag_total_loss_running_mean=RunningMean.zeros(Tag),
            tag_token_count_running_mean=RunningMean.zeros(Tag),
            tag_bytes_per_token_running_mean=RunningMean.zeros(Tag),
            per_token_loss_running_mean=RunningMean.zeros(Vocab, dtype=jnp.float32),
        )

        for batch in loading_time_tracker:
            my_batch, my_tags = batch

            # my_tags is a list of arrays, one for each example.
            # we want to stack them and add the batch axis
            my_tags = hax.stack(EvalBatch, *my_tags)
            batch = LmExample(my_batch, my_tags)

            batch = hax.shard_with_axis_mapping(batch, axis_mapping)

            _, running_means = compute_and_accumulate_loss_step(model_to_eval, running_means, batch)

            pbar.update(EvalBatch.size)

        pbar.close()

        # ok now we have the running means, we can compute the final metrics
        # we do this on the host
        (micro_loss, total_tokens) = running_means.total_loss_running_mean.get()

        (tag_losses, tag_tokens) = running_means.tag_total_loss_running_mean.get()

        (micro_loss, total_tokens, tag_losses, tag_tokens) = jax.device_get(
            (micro_loss, total_tokens, tag_losses, tag_tokens)
        )

        micro_loss = micro_loss.item() / total_tokens.item()

        tag_losses = tag_losses / tag_tokens
        tag_losses[np.isnan(tag_losses)] = 0.0

        tag_loss_dict = {tag: tag_losses[i].item() for tag, i in dataset.tag_to_index.items()}

        macro_loss = np.mean([loss for loss in tag_loss_dict.values() if loss > 0.0])

        micro_bpb, macro_bpb, tag_macro_bpb, tag_micro_bpb = _compute_bpb(
            running_means, dataset.tag_to_index, Tag, tokenizer is not None
        )

        tag_micro_losses, tag_macro_losses = _aggregate_tagged_losses(tag_loss_dict)

        # Per-token loss calculations
        per_token_loss_sum, per_token_loss_count = running_means.per_token_loss_running_mean.get()
        per_token_loss_sum, per_token_loss_count = jax.device_get((per_token_loss_sum, per_token_loss_count))

        per_token_loss_array = np.divide(
            per_token_loss_sum.array,
            per_token_loss_count.array,
            where=(per_token_loss_count.array > 0),
            out=np.zeros_like(per_token_loss_sum.array),
        )

        appeared_mask = per_token_loss_count.array > 0
        if np.any(appeared_mask):
            token_macro_loss = float(np.mean(per_token_loss_array[appeared_mask]))
        else:
            token_macro_loss = 0.0

        per_token_loss_named = hax.named(per_token_loss_array, Vocab)

        return EvalResult(
            micro_avg_loss=micro_loss,
            macro_avg_loss=macro_loss,
            tag_macro_losses=tag_macro_losses,
            tag_micro_losses=tag_micro_losses,
            total_eval_loading_time=loading_time_tracker.total_time,
            micro_bpb=micro_bpb,
            macro_bpb=macro_bpb,
            tag_macro_bpb=tag_macro_bpb,
            tag_micro_bpb=tag_micro_bpb,
            per_token_loss=per_token_loss_named,
            token_macro_loss=token_macro_loss,
        )

    def log_eval_result(result: EvalResult):
        log_dict = _construct_log_dict(result, prefix, use_ema=False)
        levanter.tracker.log_metrics(log_dict)

    return callback


def _join_prefix(prefix: str, tag: str) -> str:
    if prefix:
        return f"{prefix}/{tag}"
    return tag


def _compute_bpb(
    running_means: _EvalRunningMeans,
    tag_to_index: Mapping[str, int],
    Tag: Axis,
    has_tokenizer: bool,
) -> tuple[Optional[float], Optional[float], Optional[dict[str, float]], Optional[dict[str, float]]]:
    if not has_tokenizer:
        return None, None, None, None

    (total_bytes, total_tokens) = running_means.bytes_per_token_running_mean.get()
    (tag_bytes, tag_tokens) = running_means.tag_bytes_per_token_running_mean.get()

    (total_bytes, total_tokens, tag_bytes, tag_tokens) = jax.device_get(
        (total_bytes, total_tokens, tag_bytes, tag_tokens)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        micro_bpb = (total_bytes / total_tokens).item() / np.log(2)

        tag_bpb = (tag_bytes / tag_tokens) / np.log(2)
        tag_bpb[np.isnan(tag_bpb)] = 0.0

        tag_bpb_dict = {tag: tag_bpb[i].item() for tag, i in tag_to_index.items()}

        tag_micro_bpb, tag_macro_bpb = _aggregate_tagged_losses(tag_bpb_dict)

        macro_bpb = np.mean([b for b in tag_bpb_dict.values() if b > 0.0])

    return micro_bpb, macro_bpb, tag_macro_bpb, tag_micro_bpb


def _aggregate_tagged_losses(tag_losses: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate losses by tag hierarchy."""
    tag_micro_losses = defaultdict(float)
    tag_macro_losses = defaultdict(list)

    for tag, loss in tag_losses.items():
        parts = tag.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            tag_micro_losses[prefix] += loss
            tag_macro_losses[prefix].append(loss)

    # now average the macro losses
    tag_macro_losses_avg = {tag: np.mean(losses) for tag, losses in tag_macro_losses.items()}

    return dict(tag_micro_losses), tag_macro_losses_avg


def _construct_log_dict(result: EvalResult, prefix: str, use_ema: bool) -> dict[str, Arrayish]:
    log_dict = {}
    if use_ema:
        prefix = f"ema/{prefix}"

    log_dict[_join_prefix(prefix, "loss")] = result.micro_avg_loss
    if result.micro_bpb is not None:
        log_dict[_join_prefix(prefix, "bpb")] = result.micro_bpb

    log_dict[_join_prefix(prefix, "macro_loss")] = result.macro_avg_loss
    if result.macro_bpb is not None:
        log_dict[_join_prefix(prefix, "macro_bpb")] = result.macro_bpb

    if result.token_macro_loss is not None:
        log_dict[_join_prefix(prefix, "token_macro_loss")] = result.token_macro_loss

    for tag, loss in result.tag_macro_losses.items():
        log_dict[_join_prefix(f"{prefix}/macro_loss", tag)] = loss
        if result.tag_macro_bpb is not None and tag in result.tag_macro_bpb:
            log_dict[_join_prefix(f"{prefix}/macro_bpb", tag)] = result.tag_macro_bpb[tag]

    for tag, loss in result.tag_micro_losses.items():
        log_dict[_join_prefix(f"{prefix}/micro_loss", tag)] = loss
        if result.tag_micro_bpb is not None and tag in result.tag_micro_bpb:
            log_dict[_join_prefix(f"{prefix}/micro_bpb", tag)] = result.tag_micro_bpb[tag]

    return log_dict