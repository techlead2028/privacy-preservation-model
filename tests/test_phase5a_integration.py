"""
Phase 5A Integration Tests: LoRADeltaClient → PrivacyEngine → Flower pipeline.

These tests validate that Member 2's ΔLoRA can successfully pass through
Member 3's PrivacyEngine with DP disabled and masking disabled (pass-through mode).

All tests use SYNTHETIC mode (no real TinyLlama SFT).
"""

import pytest
import torch
import numpy as np
from privacy.privacy_engine import PrivacyEngine
from tests.synthetic_lora_client import (
    SyntheticLoRADeltaClient,
    LORA_PARAM_KEYS,
    _get_shape_for_key,
)


@pytest.fixture
def lora_client():
    """A deterministic synthetic LoRA client."""
    return SyntheticLoRADeltaClient(seed=42, num_examples=500)


@pytest.fixture
def engine():
    return PrivacyEngine()


# ─────────────────────────────────────────────────────────
# A. LoRADeltaClient produces ΔLoRA
# ─────────────────────────────────────────────────────────

def test_lora_client_produces_delta(lora_client):
    delta, num_examples, metrics = lora_client.compute_delta()

    assert isinstance(delta, dict)
    assert len(delta) == 88
    for key, tensor in delta.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")


# ─────────────────────────────────────────────────────────
# B. ΔLoRA can be passed through PrivacyEngine
# ─────────────────────────────────────────────────────────

def test_delta_passes_through_privacy_engine(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()

    # Phase 5A: DP disabled, mask disabled
    config = {"enable_dp": False}
    result = engine.process_update(delta, num_examples=num_examples, config=config)

    assert isinstance(result, list)
    assert len(result) == 88


# ─────────────────────────────────────────────────────────
# C. Number of tensors remains 88
# ─────────────────────────────────────────────────────────

def test_tensor_count_is_88(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()
    result = engine.process_update(delta, config={"enable_dp": False})
    assert len(result) == 88


# ─────────────────────────────────────────────────────────
# D. Tensor shapes remain unchanged
# ─────────────────────────────────────────────────────────

def test_shapes_preserved(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()
    result = engine.process_update(delta, config={"enable_dp": False})

    sorted_keys = sorted(delta.keys())
    for i, key in enumerate(sorted_keys):
        expected_shape = _get_shape_for_key(key)
        assert result[i].shape == expected_shape, (
            f"Shape mismatch for {key}: expected {expected_shape}, got {result[i].shape}"
        )


# ─────────────────────────────────────────────────────────
# E. dtype remains float32
# ─────────────────────────────────────────────────────────

def test_dtype_float32(lora_client, engine):
    delta, _, _ = lora_client.compute_delta()
    result = engine.process_update(delta, config={"enable_dp": False})

    for i, arr in enumerate(result):
        assert arr.dtype == np.float32, f"Tensor {i} has dtype {arr.dtype}"


# ─────────────────────────────────────────────────────────
# F. Parameter ordering remains unchanged
# ─────────────────────────────────────────────────────────

def test_parameter_ordering_preserved(lora_client, engine):
    delta, _, _ = lora_client.compute_delta()
    result = engine.process_update(delta, config={"enable_dp": False})

    # PrivacyEngine sorts keys; verify that the Flower list
    # matches the existing deterministic sorted-key order from Member 2
    delta_list, _, _ = lora_client.compute_delta_as_list()

    assert len(result) == len(delta_list)
    for i in range(len(result)):
        np.testing.assert_allclose(result[i], delta_list[i], atol=1e-6)


# ─────────────────────────────────────────────────────────
# G. num_examples remains unchanged
# ─────────────────────────────────────────────────────────

def test_num_examples_unchanged(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()
    # PrivacyEngine does not modify num_examples — it passes through
    result = engine.process_update(delta, num_examples=num_examples, config={"enable_dp": False})

    # num_examples is metadata, not modified by PrivacyEngine
    assert num_examples == 500


# ─────────────────────────────────────────────────────────
# H. Training metrics remain unchanged
# ─────────────────────────────────────────────────────────

def test_metrics_unchanged(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()
    engine.process_update(delta, num_examples=num_examples, config={"enable_dp": False})

    # Metrics are metadata, not modified by PrivacyEngine
    assert metrics["train_loss"] == 2.34
    assert metrics["learning_rate"] == 2e-4
    assert metrics["epoch"] == 1


# ─────────────────────────────────────────────────────────
# I. PrivacyEngine does not mutate the original ΔLoRA
# ─────────────────────────────────────────────────────────

def test_input_not_mutated(lora_client, engine):
    delta, num_examples, metrics = lora_client.compute_delta()
    original_copy = {k: v.clone() for k, v in delta.items()}

    engine.process_update(delta, num_examples=num_examples, config={"enable_dp": False})

    for k in delta:
        assert torch.equal(delta[k], original_copy[k]), f"Tensor {k} was mutated"


# ─────────────────────────────────────────────────────────
# J. Flower client returns protected_deltas, num_examples, metrics
# ─────────────────────────────────────────────────────────

def test_flower_return_contract(lora_client, engine):
    """Simulate the full Flower fit() return."""
    delta, num_examples, metrics = lora_client.compute_delta()
    protected_deltas = engine.process_update(delta, num_examples=num_examples, config={"enable_dp": False})

    # This is what Flower expects from fit()
    assert isinstance(protected_deltas, list)
    assert isinstance(num_examples, int)
    assert isinstance(metrics, dict)

    # All tensors are np.ndarray
    for arr in protected_deltas:
        assert isinstance(arr, np.ndarray)


# ─────────────────────────────────────────────────────────
# K. With DP/masking disabled, values equal original ΔLoRA
# ─────────────────────────────────────────────────────────

def test_passthrough_numerical_equivalence(lora_client, engine):
    delta, _, _ = lora_client.compute_delta()
    result = engine.process_update(delta, config={"enable_dp": False})

    sorted_keys = sorted(delta.keys())
    for i, key in enumerate(sorted_keys):
        np.testing.assert_allclose(
            result[i],
            delta[key].numpy(),
            atol=1e-6,
            err_msg=f"Value mismatch for {key}",
        )
