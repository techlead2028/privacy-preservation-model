"""
Synthetic LoRADeltaClient — simulates Member 2's contract for integration testing.

Member 2's real LoRADeltaClient:
  1. Receives global LoRA parameters.
  2. Performs local TinyLlama + LoRA SFT.
  3. Extracts the updated local LoRA parameters.
  4. Computes ΔLoRA = local_LoRA - global_LoRA.
  5. Returns (List[np.ndarray], num_examples, metrics).

This synthetic version produces a deterministic 88-tensor ΔLoRA update
matching the real TinyLlama LoRA adapter structure (22 transformer layers × 
4 projections: q_proj, k_proj, v_proj, o_proj × lora_A + lora_B = 2 matrices each).

22 layers × 4 projections × 2 (A + B) = 176? No — TinyLlama uses rank-8 LoRA
on q_proj and v_proj only across 22 layers:
22 layers × 2 projections × 2 matrices = 88 tensors.

Shapes (TinyLlama 1.1B with rank=8, hidden_dim=2048):
  lora_A.weight: (8, 2048)
  lora_B.weight: (2048, 8)

This module is for TESTING ONLY and does not contain real SFT logic.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any


# TinyLlama LoRA configuration
NUM_LAYERS = 22
LORA_RANK = 8
HIDDEN_DIM = 2048
PROJECTIONS = ["q_proj", "v_proj"]  # Standard LoRA targets


def _build_lora_param_keys() -> List[str]:
    """Build the 88 deterministic sorted parameter keys matching TinyLlama LoRA."""
    keys = []
    for layer_idx in range(NUM_LAYERS):
        for proj in PROJECTIONS:
            keys.append(
                f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}.lora_A.weight"
            )
            keys.append(
                f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}.lora_B.weight"
            )
    return sorted(keys)


LORA_PARAM_KEYS = _build_lora_param_keys()
assert len(LORA_PARAM_KEYS) == 88, f"Expected 88 keys, got {len(LORA_PARAM_KEYS)}"


def _get_shape_for_key(key: str) -> Tuple[int, int]:
    """Return the tensor shape based on whether this is a lora_A or lora_B weight."""
    if "lora_A.weight" in key:
        return (LORA_RANK, HIDDEN_DIM)
    elif "lora_B.weight" in key:
        return (HIDDEN_DIM, LORA_RANK)
    else:
        raise ValueError(f"Unknown LoRA key pattern: {key}")


class SyntheticLoRADeltaClient:
    """
    Simulates Member 2's LoRADeltaClient contract.
    
    Produces a deterministic 88-tensor ΔLoRA as Dict[str, torch.Tensor],
    plus num_examples and training metrics — exactly matching the interface
    that a real LoRADeltaClient would produce after local SFT.
    """

    def __init__(self, seed: int = 42, num_examples: int = 500):
        self.seed = seed
        self.num_examples = num_examples

    def compute_delta(self) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        """
        Simulate the ΔLoRA computation.
        
        Returns:
            delta: Dict[str, torch.Tensor] — 88 tensors, float32, CPU
            num_examples: int — number of training examples used
            metrics: dict — training metrics
        """
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        delta: Dict[str, torch.Tensor] = {}
        for key in LORA_PARAM_KEYS:
            shape = _get_shape_for_key(key)
            # Small random values simulating a realistic ΔLoRA magnitude
            tensor = torch.randn(shape, generator=rng, dtype=torch.float32) * 0.01
            delta[key] = tensor

        metrics = {
            "train_loss": 2.34,
            "learning_rate": 2e-4,
            "epoch": 1,
        }

        return delta, self.num_examples, metrics

    def compute_delta_as_list(self) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Simulate the full LoRADeltaClient.fit() return value.
        
        This is what Member 2 currently returns to Flower:
        List[np.ndarray] in deterministic sorted-key order.
        """
        delta_dict, num_examples, metrics = self.compute_delta()
        
        sorted_keys = sorted(delta_dict.keys())
        delta_list = [delta_dict[k].numpy() for k in sorted_keys]
        
        return delta_list, num_examples, metrics
