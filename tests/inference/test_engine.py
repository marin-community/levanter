# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax.numpy as jnp
import pytest

from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.decode_state import SeqDecodingParams
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig


@pytest.fixture(scope="module")
def trainer_config():
    return TrainerConfig(model_axis_size=1)


@pytest.fixture(scope="module")
def baby_llama_model(trainer_config):
    """Load the baby llama model and tokenizer."""
    hf_checkpoint = "timinar/baby-llama-58m"
    model_config = LlamaConfig()
    tokenizer = load_tokenizer(hf_checkpoint)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = HFCheckpointConverter(
            LlamaConfig,
            reference_checkpoint=hf_checkpoint,
            tokenizer=tokenizer,
        )

        model = converter.load_pretrained(
            model_config.model_type,
            ref=hf_checkpoint,
            dtype=trainer_config.mp.compute_dtype,
            axis_mapping=trainer_config.parameter_axis_mapping,
        )

    return model, tokenizer


def test_deterministic_generation_with_temp_zero(baby_llama_model, trainer_config):
    """Test that generation with temperature=0 produces deterministic, exact output."""
    model, tokenizer = baby_llama_model

    # Create engine config with small settings for testing
    config = InferenceEngineConfig(
        max_seq_len=32,
        max_seqs=4,
        page_size=8,
        tokens_per_round=16,
    )

    # Build engine
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config)

        # Test prompt
        prompt = "Hello world"
        prompt_tokens = tokenizer.encode(prompt)

        # Create request with temp=0
        decode_params = SeqDecodingParams(
            max_num_tokens=jnp.array(10, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jnp.array([0, 0], dtype=jnp.uint32),
        )

        request = Request(
            prompt_tokens=prompt_tokens,
            request_id=0,
            decode_params=decode_params,
            n_generations=1,
        )

        # Generate
        result = engine.generate([request])

        # Expected tokens captured from running with temperature=0
        # Prompt: "Hello world" -> tokens [13536, 963]
        # With temp=0, this should be deterministic
        expected_tokens = [269, 14, 169, 238, 208, 696, 818, 16, 78, 14, 169, 4239, 169, 43, 43, 444]

        # Validate exact match
        assert len(result.tokens) == 1, "Should have exactly 1 generation"
        assert result.tokens[0] == expected_tokens, f"Expected {expected_tokens}, got {result.tokens[0]}"
        assert len(result.tokens[0]) == len(expected_tokens), "Token count mismatch"

        # Validate decoded text matches
        decoded = tokenizer.decode(result.tokens[0])
        expected_text = " is,\n of a after great.l,\nBut\nII'm"
        assert decoded == expected_text, f"Expected '{expected_text}', got '{decoded}'"


def test_multiple_generations(baby_llama_model, trainer_config):
    """Test that n_generations produces correct number of outputs."""
    model, tokenizer = baby_llama_model

    config = InferenceEngineConfig(
        max_seq_len=32,
        max_seqs=4,
        page_size=8,
        tokens_per_round=16,
    )

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config)

        prompt = "Hello world"
        prompt_tokens = tokenizer.encode(prompt)

        decode_params = SeqDecodingParams(
            max_num_tokens=jnp.array(10, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jnp.array([0, 0], dtype=jnp.uint32),
        )

        request = Request(
            prompt_tokens=prompt_tokens,
            request_id=0,
            decode_params=decode_params,
            n_generations=4,
        )

        result = engine.generate([request])

        # Validate we got 4 generations
        assert len(result.tokens) == 4, f"Expected 4 generations, got {len(result.tokens)}"

        # With temp=0, all generations should be identical
        for i in range(1, 4):
            assert result.tokens[i] == result.tokens[0], f"Generation {i} differs from generation 0 with temp=0"
