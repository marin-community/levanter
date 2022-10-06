import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
from equinox import filter_vmap
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec

import haliax as hax
from haliax import Axis
from haliax.partitioning import (
    ResourceAxis,
    axis_mapping,
    infer_resource_partitions,
    named_pjit,
    round_axis_for_partitioning,
)
from levanter import callbacks
from levanter.callbacks import log_performance_stats, log_to_wandb, pbar_logger, wandb_xla_logger
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.logging import capture_time, log_time_to_wandb
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import pyrallis
from transformers import GPT2Tokenizer

import wandb
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig
from levanter.jax_utils import global_key_array, parameter_count
from levanter.modeling_utils import accumulate_gradients_sharded
from levanter.trainer_hooks import StepInfo, TrainerHooks


logger = logging.getLogger(__name__)


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    log_z_regularization: float = 0.0


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.trainer.initialize(config)

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer
    dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("train"),
        config.trainer.train_mesh_info,
        config.model.seq_len,
    )

    eval_dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("validation"),
        config.trainer.eval_mesh_info,
        config.model.seq_len,
    )

    with config.trainer.device_mesh as mesh, axis_mapping(config.trainer.axis_resources):

        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        mp = config.trainer.mp
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size))
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # initialize the model and optimizer, and convert to appropriate dtype
        # doing this in a pjit means that the model and optimizer states are already sharded
        # TODO: think about how we want to do this if we want to load a checkpoint
        with axis_mapping(config.trainer.parameter_axis_resources, merge=True):
            optim = config.trainer.optimizer()

            @named_pjit
            def init_state():
                model = mp.cast_to_param(Gpt2LMHeadModel(Vocab, config.model, key=model_key))
                opt_state = optim.init(model)
                return model, opt_state

            model, opt_state = init_state()
            opt_state_resources = infer_resource_partitions(opt_state)
            model_resources = infer_resource_partitions(model)

        # log some info about the model
        wandb.summary["parameter_count"] = parameter_count(model)

        # loss function
        def compute_loss(model: Gpt2LMHeadModel, input_ids, key, inference):
            pred_y = model(input_ids, inference=inference, key=key)
            pred_y = mp.cast_to_output(pred_y)

            # TODO: would prefer to do this in haliax name land, but it's not clear how to do that
            # could add a where mask which is pretty normal
            pred_y = pred_y[:-1]
            target_y = input_ids[1:]
            labels = jax.nn.one_hot(target_y, Vocab.size)

            log_normalizers = jax.nn.logsumexp(pred_y, -1, keepdims=True)
            log_normalized = pred_y - log_normalizers

            loss = -jnp.sum(labels * log_normalized, axis=-1)
            loss = jnp.mean(loss)

            if not inference and config.log_z_regularization > 0:
                logz_mse = jnp.mean((log_normalizers**2))
                loss += config.log_z_regularization * logz_mse

            return loss

        def mean_loss(model: Gpt2LMHeadModel, input_ids, key, inference):
            # None here means the first argument (the model) is not vectorized but instead broadcasted
            compute_loss_vmap = filter_vmap(compute_loss, args=(None,), spmd_axis_name=ResourceAxis.DATA)
            return jnp.mean(compute_loss_vmap(model, input_ids, key, inference))

        compute_loss_pjit = pjit(
            partial(mean_loss, inference=True, key=None),
            in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
            out_axis_resources=None,
        )

        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_value_and_grad(partial(mean_loss, inference=False))

        # boilerplate hooks and such
        engine = TrainerHooks()
        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(log_performance_stats(config.model.seq_len, config.trainer.train_batch_size), every=1)

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch,)

        evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)
        engine.add_hook(evaluate, every=config.trainer.steps_per_eval)
        save = callbacks.save_model(config.trainer.checkpoint_path)
        engine.add_hook(save, every=config.trainer.steps_per_save)

        # a bit hacky, but we'd prefer this go after eval, so that we can capture the xla dumps for eval too
        engine.add_hook(wandb_xla_logger(config.trainer.wandb), every=1000)

        # data loader
        iter_data = iter(dataset)

        # load the last checkpoint and resume if we want
        # TODO: wandb resume logic?
        resume_step = None
        if config.trainer.load_last_checkpoint:
            with jax.default_device(jax.devices("cpu")[0]):
                checkpoint = load_checkpoint(
                    model,
                    (opt_state, training_key),
                    config.trainer.load_checkpoint_path or config.trainer.checkpoint_path,
                )
            if checkpoint is not None:
                model, (opt_state, training_key), resume_step = checkpoint
            elif config.trainer.load_checkpoint_path:
                raise ValueError("No checkpoint found")
            else:
                logger.info("No checkpoint found. Starting from scratch")

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(resume_step + 1), desc="seeking data"):
                next(iter_data)
            resume_step = resume_step + 1
        else:
            resume_step = 0

        mesh_info = config.trainer.train_mesh_info

        def train_step(model, opt_state, input_ids, keys):
            model_inf = mp.cast_to_compute(model)
            with axis_mapping(config.trainer.axis_resources, merge=False):
                model_inf = hax.logically_sharded(model_inf)

                loss, grads = accumulate_gradients_sharded(
                    compute_loss_and_grad,
                    mesh_info.data_axis_size,
                    mesh_info.per_device_parallelism,
                    model_inf,
                    input_ids,
                    keys,
                )

            with jax.named_scope("optimizer"):
                with axis_mapping(config.trainer.parameter_axis_resources, merge=True):
                    updates, opt_state = optim.update(grads, opt_state, params=model)
                    model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # keys are sharded in the same way as input_ids
        # TODO: maybe put keys in the data iterator?
        data_resources = dataset.partition_spec

        train_step = pjit(
            train_step,
            in_axis_resources=(
                model_resources,
                opt_state_resources,
                data_resources,
                data_resources,
            ),
            out_axis_resources=(None, model_resources, opt_state_resources),
            donate_argnums=(0, 1),
        )

        for step in range(resume_step, config.trainer.num_train_steps):
            with capture_time() as step_time:

                with log_time_to_wandb("throughput/loading_time", step=step):
                    input_ids = next(iter_data)
                    my_key, training_key = jrandom.split(training_key, 2)
                    micro_keys = global_key_array(my_key, input_ids.shape[:-1], mesh, dataset.partition_spec[:-1])

                step_loss, model, opt_state = train_step(model, opt_state, input_ids, micro_keys)
                step_loss = jnp.mean(step_loss).item()

            engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        last_step = StepInfo(
            config.trainer.num_train_steps,
            model,
            opt_state,
            step_loss,
            training_key,
            step_duration=step_time(),
        )

        evaluate(last_step)
        save(last_step)


if __name__ == "__main__":
    main()
