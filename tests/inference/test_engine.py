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
        expected_tokens = [16, 169, 16, 169, 43, 43, 444, 79, 28, 170, 170, 169, 43, 444, 635, 16]

        # Validate exact match
        assert len(result.tokens) == 1, "Should have exactly 1 generation"
        assert result.tokens[0] == expected_tokens, f"Expected {expected_tokens}, got {result.tokens[0]}"
        assert len(result.tokens[0]) == len(expected_tokens), "Token count mismatch"

        # Validate decoded text matches
        decoded = tokenizer.decode(result.tokens[0])
        expected_text = ".\n.\nII'mm:  \nI'm am."
        assert decoded == expected_text, f"Expected '{expected_text}', got '{decoded}'"


def test_one_seq_two_generations(baby_llama_model, trainer_config):
    """Test 1 sequence with 2 n_generations."""
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
            n_generations=2,
        )

        result = engine.generate([request])

        # Validate we got 2 generations
        assert len(result.tokens) == 2, f"Expected 2 generations, got {len(result.tokens)}"

        # With temp=0, all generations should be identical
        assert result.tokens[1] == result.tokens[0], "Generation 1 differs from generation 0 with temp=0"

        # Should match the single generation output
        expected_tokens = [16, 169, 16, 169, 43, 43, 444, 79, 28, 170, 170, 169, 43, 444, 635, 16]
        assert result.tokens[0] == expected_tokens, f"Expected {expected_tokens}, got {result.tokens[0]}"


def test_two_seqs_one_generation(baby_llama_model, trainer_config):
    """Test 2 sequences with 1 n_generation each."""
    model, tokenizer = baby_llama_model

    config = InferenceEngineConfig(
        max_seq_len=32,
        max_seqs=4,
        page_size=8,
        tokens_per_round=16,
    )

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config)

        prompt1 = "Hello world"
        prompt2 = "Goodbye"
        prompt_tokens1 = tokenizer.encode(prompt1)
        prompt_tokens2 = tokenizer.encode(prompt2)

        decode_params = SeqDecodingParams(
            max_num_tokens=jnp.array(10, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jnp.array([0, 0], dtype=jnp.uint32),
        )

        request1 = Request(
            prompt_tokens=prompt_tokens1,
            request_id=0,
            decode_params=decode_params,
            n_generations=1,
        )

        request2 = Request(
            prompt_tokens=prompt_tokens2,
            request_id=1,
            decode_params=decode_params,
            n_generations=1,
        )

        result = engine.generate([request1, request2])

        # Validate we got 2 generations (1 for each sequence)
        assert len(result.tokens) == 2, f"Expected 2 generations, got {len(result.tokens)}"

        # Each should have generated tokens
        assert len(result.tokens[0]) > 0, "Sequence 0 should have generated tokens"
        assert len(result.tokens[1]) > 0, "Sequence 1 should have generated tokens"


def test_two_seqs_two_generations(baby_llama_model, trainer_config):
    """Test 2 sequences with 2 n_generations each."""
    model, tokenizer = baby_llama_model

    config = InferenceEngineConfig(
        max_seq_len=32,
        max_seqs=8,  # Need more slots for 2 seqs * 2 generations
        page_size=8,
        tokens_per_round=16,
    )

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config)

        prompt1 = "Hello world"
        prompt2 = "Goodbye"
        prompt_tokens1 = tokenizer.encode(prompt1)
        prompt_tokens2 = tokenizer.encode(prompt2)

        decode_params = SeqDecodingParams(
            max_num_tokens=jnp.array(10, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jnp.array([0, 0], dtype=jnp.uint32),
        )

        request1 = Request(
            prompt_tokens=prompt_tokens1,
            request_id=0,
            decode_params=decode_params,
            n_generations=2,
        )

        request2 = Request(
            prompt_tokens=prompt_tokens2,
            request_id=1,
            decode_params=decode_params,
            n_generations=2,
        )

        result = engine.generate([request1, request2])

        # Validate we got 4 generations (2 for each sequence)
        assert len(result.tokens) == 4, f"Expected 4 generations, got {len(result.tokens)}"

        # With temp=0, generations for the same request should be identical
        assert result.tokens[1] == result.tokens[0], "Request 1 generations should match with temp=0"
        assert result.tokens[3] == result.tokens[2], "Request 2 generations should match with temp=0"
