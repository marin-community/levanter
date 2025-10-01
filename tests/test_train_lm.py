# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Callable

import jax
import pytest

from haliax.quantization import QuantizationConfig

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.distributed import DistributedConfig, RayConfig
from levanter.callbacks import StepInfo
from levanter.eval import EvalPlugin, EvalPluginConfig
from levanter.tracker import NoopConfig


def _create_test_model_config() -> train_lm.LlamaConfig:
    """Create a small test model configuration for testing."""
    return train_lm.LlamaConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        seq_len=64,
        hidden_dim=32,
        attn_backend=None,  # use default for platform
    )


def _create_test_trainer_config(**kwargs: Any) -> train_lm.TrainerConfig:
    """Create a test trainer configuration."""
    config = {
        "num_train_steps": 2,
        "train_batch_size": len(jax.devices()),
        "max_eval_batches": 1,
        "tracker": NoopConfig(),
        "require_accelerator": False,
        "ray": RayConfig(auto_start_cluster=False),
        "distributed": DistributedConfig(initialize_jax_distributed=False),
        **kwargs,
    }
    return train_lm.TrainerConfig(**config)


def _cleanup_wandb():
    """Clean up wandb files after test."""
    try:
        os.unlink("wandb")
    except Exception:
        pass


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=_create_test_model_config(),
                trainer=_create_test_trainer_config(),
            )
            train_lm.main(config)
        finally:
            _cleanup_wandb()


@pytest.mark.entry
def test_train_lm_fp8():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=_create_test_model_config(),
                trainer=_create_test_trainer_config(quantization=QuantizationConfig(fp8=True)),
            )
            train_lm.main(config)
        finally:
            _cleanup_wandb()


class TestTrainingEvalPlugin(EvalPlugin):
    context = dict(called=False)

    def create_callback(self, *args: Any, **kwargs: Any) -> Callable[[StepInfo], None]:
        def callback(step_info: StepInfo) -> None:
            TestTrainingEvalPlugin.context["called"] = True

        return callback


@pytest.mark.entry
def test_train_lm_with_eval_plugin():
    # make sure plugins can be loaded and executed successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            cls = TestTrainingEvalPlugin
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=_create_test_model_config(),
                trainer=_create_test_trainer_config(steps_per_eval=1),  # Eval every step to ensure plugin is called
                eval_plugins=[
                    EvalPluginConfig(
                        plugin_class=f"{cls.__module__}.{cls.__name__}",
                        steps=1,
                    )
                ],
            )
            train_lm.main(config)
            assert TestTrainingEvalPlugin.context["called"], "Plugin callback was not called during training"
        finally:
            _cleanup_wandb()
