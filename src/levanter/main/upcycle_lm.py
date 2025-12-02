import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Generic, List, TypeVar, Union

import equinox as eqx
import jax
from lenses import lens

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.models.llama import LlamaConfig
from levanter.models.mixtral import MixtralConfig
from levanter.trainer import TrainerConfig


logger = logging.getLogger(__name__)
M = TypeVar("M")


@dataclass
class UpcycleLmConfig:
    checkpoint_paths: List[str]
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    dense_model: LlamaConfig = field(default_factory=LlamaConfig)
    sparse_model: MixtralConfig = field(default_factory=MixtralConfig)

    # the std of noise added to expert layers
    noise_scale: float = 1e-3


class DummyState(eqx.Module, Generic[M]):
    """
    This class is to enforce similar structure as TrainerState when saving checkpoint.
    """

    model: M

    @classmethod
    def init(cls, model):
        return cls(model)


def main(config: UpcycleLmConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    # how many experts do we get/split from a single checkpoint
    granularity = config.dense_model.intermediate_dim // config.sparse_model.intermediate_dim
    if config.dense_model.intermediate_dim % config.sparse_model.intermediate_dim != 0 or granularity < 1:
        raise ValueError("Sparse model's intermediate size should be a fraction of dense model's intermediate size.")
    experts_per_checkpoint = config.sparse_model.n_routed_experts // len(config.checkpoint_paths)
    # how many times a checkpoint is replicated in the upcycled model
    expansion = experts_per_checkpoint // granularity

    print(f"Upcycle configuration: {expansion=}, {granularity=}")

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    if len(config.checkpoint_paths) == 0:
        raise ValueError("Must specify at least 1 dense checkpoint path to upcycle.")

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        sparse_model = hax.named_jit(config.sparse_model.build, axis_resources=parameter_axis_mapping)(Vocab, key=key)
        sparse_mlp = sparse_model.transformer.layers.stacked.block_sparse_moe.experts
        avg_model = None

        # initialize the model
        for i, checkpoint_path in enumerate(config.checkpoint_paths):
            print(f"Working on the {i}-th checkpoint.")
            # initialize the model
            dense_model = eqx.filter_eval_shape(config.dense_model.build, Vocab, key=key)
            dense_model = load_checkpoint(
                dense_model, checkpoint_path, subpath="model", axis_mapping=parameter_axis_mapping
            )

            # We want use the average weights for all non-moe layers (e.g. attention, layernorm).
            if avg_model is None:
                avg_model = jax.tree.map(lambda x: x / len(config.checkpoint_paths), dense_model)
            else:
                avg_model = jax.tree.map(lambda x, avg: avg + x / len(config.checkpoint_paths), dense_model, avg_model)

            dense_mlp = dense_model.transformer.layers.stacked.mlp
            gate_proj = dense_mlp.gate_proj.weight
            up_proj = dense_mlp.up_proj.weight
            down_proj = dense_mlp.down_proj.weight

            if granularity > 1:
                SplitExperts = hax.Axis("experts", granularity)
                gate_proj = hax.unflatten_axis(
                    gate_proj, config.dense_model.Mlp, (SplitExperts, config.sparse_model.Mlp)
                )
                up_proj = hax.unflatten_axis(up_proj, config.dense_model.Mlp, (SplitExperts, config.sparse_model.Mlp))
                down_proj = hax.unflatten_axis(
                    down_proj, config.dense_model.Mlp, (SplitExperts, config.sparse_model.Mlp)
                )

            w1 = sparse_mlp.w1.weight
            w2 = sparse_mlp.w2.weight
            w3 = sparse_mlp.w3.weight
            for j in range(expansion):
                start = i * experts_per_checkpoint + j * granularity
                w1 = w1.at[{"experts": slice(start, start + granularity)}].set(gate_proj)
                w2 = w2.at[{"experts": slice(start, start + granularity)}].set(down_proj)
                w3 = w3.at[{"experts": slice(start, start + granularity)}].set(up_proj)

            key, w1_key, w2_key, w3_key = jax.random.split(key, 4)

            # Add tiny noise to moe layers.
            experts = sparse_model.transformer.layers.stacked.block_sparse_moe.experts
            experts = lens.w1.weight.set(w1 + hax.random.normal(w1_key, w1.axes) * config.noise_scale)(experts)
            experts = lens.w2.weight.set(w2 + hax.random.normal(w2_key, w2.axes) * config.noise_scale)(experts)
            experts = lens.w3.weight.set(w3 + hax.random.normal(w3_key, w3.axes) * config.noise_scale)(experts)
            sparse_model = lens.transformer.layers.stacked.block_sparse_moe.experts.set(experts)(sparse_model)

        if avg_model is not None:
            sparse_model = lens.embeddings.set(avg_model.embeddings)(sparse_model)

            new_self_attn = dataclasses.replace(
                avg_model.transformer.layers.stacked.self_attn, config=config.sparse_model
            )
            sparse_model = lens.transformer.layers.stacked.self_attn.set(new_self_attn)(sparse_model)
            sparse_model = lens.transformer.layers.stacked.input_layernorm.set(
                avg_model.transformer.layers.stacked.input_layernorm
            )(sparse_model)
            sparse_model = lens.transformer.layers.stacked.post_attention_layernorm.set(
                avg_model.transformer.layers.stacked.post_attention_layernorm
            )(sparse_model)
            sparse_model = lens.transformer.norm.set(avg_model.transformer.norm)(sparse_model)

            sparse_model = lens.lm_head.set(avg_model.lm_head)(sparse_model)
        state: DummyState = DummyState(sparse_model)

        save_checkpoint(state, step=0, checkpoint_path=config.trainer.checkpointer.base_path, is_temporary=False)

    # ray tasks don't reliably wait for the subprocesses to finish, so we need to manually finish the tracker
    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
