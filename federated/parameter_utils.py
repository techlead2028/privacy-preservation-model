import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import flwr.common as flwr_common


def lora_dict_to_ndarrays(lora_dict: Dict[str, torch.Tensor]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Deterministically converts a dictionary of LoRA PyTorch tensors to a list of
    NumPy arrays suitable for Flower communication, alongside sorted parameter keys.

    Args:
        lora_dict: Mapping from parameter name to torch.Tensor.

    Returns:
        Tuple[List[np.ndarray], List[str]]:
            - ndarrays: List of contiguous float32 NumPy arrays in deterministic key order.
            - keys: Sorted list of parameter names.
    """
    sorted_keys = sorted(lora_dict.keys())
    ndarrays: List[np.ndarray] = []
    for k in sorted_keys:
        t = lora_dict[k]
        # Ensure detached, CPU-resident, contiguous float32 numpy array
        arr = t.detach().cpu().to(torch.float32).contiguous().numpy()
        ndarrays.append(arr)
    return ndarrays, sorted_keys


def ndarrays_to_lora_dict(ndarrays: List[np.ndarray], keys: List[str]) -> Dict[str, torch.Tensor]:
    """
    Deterministically converts a list of Flower NumPy arrays back to a PyTorch LoRA
    parameter dictionary using the provided sorted keys.

    Args:
        ndarrays: List of NumPy arrays in matching key order.
        keys: List of parameter names corresponding to the arrays.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to CPU float32 tensors.
    """
    if len(ndarrays) != len(keys):
        raise ValueError(
            f"Mismatched lengths: {len(ndarrays)} ndarrays vs {len(keys)} keys."
        )
    lora_dict: Dict[str, torch.Tensor] = {}
    for k, arr in zip(keys, ndarrays):
        # Convert copy to torch.float32 tensor on CPU
        tensor = torch.from_numpy(arr.copy()).to(torch.float32).contiguous()
        lora_dict[k] = tensor
    return lora_dict


def ndarrays_to_flower_parameters(ndarrays: List[np.ndarray]) -> flwr_common.Parameters:
    """
    Wraps a list of NumPy arrays into Flower's Parameters object.
    """
    return flwr_common.ndarrays_to_parameters(ndarrays)


def flower_parameters_to_ndarrays(parameters: flwr_common.Parameters) -> List[np.ndarray]:
    """
    Extracts a list of NumPy arrays from Flower's Parameters object.
    """
    return flwr_common.parameters_to_ndarrays(parameters)


def compute_lora_delta(
    local_lora: Dict[str, torch.Tensor],
    global_lora: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Computes delta updates: delta = local_lora - global_lora.
    """
    delta_dict: Dict[str, torch.Tensor] = {}
    for k, local_tensor in local_lora.items():
        if k not in global_lora:
            raise KeyError(f"Parameter key '{k}' missing from global LoRA dictionary.")
        global_tensor = global_lora[k]
        delta_dict[k] = (local_tensor.detach().cpu().to(torch.float32) -
                         global_tensor.detach().cpu().to(torch.float32)).contiguous()
    return delta_dict


def apply_lora_delta(
    global_lora: Dict[str, torch.Tensor],
    delta_lora: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Applies aggregated delta updates to global parameters:
    new_global = global_lora + delta_lora.
    """
    updated_dict: Dict[str, torch.Tensor] = {}
    for k, global_tensor in global_lora.items():
        if k not in delta_lora:
            raise KeyError(f"Parameter key '{k}' missing from delta LoRA dictionary.")
        delta_tensor = delta_lora[k]
        updated_dict[k] = (global_tensor.detach().cpu().to(torch.float32) +
                           delta_tensor.detach().cpu().to(torch.float32)).contiguous()
    return updated_dict


def compute_weighted_delta_average(
    client_deltas: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Computes weighted average of client delta ndarrays:
    aggregated_delta = sum((n_i / N_total) * delta_i for each client i)

    Args:
        client_deltas: List of tuples (delta_ndarrays, num_examples).

    Returns:
        List[np.ndarray]: Weighted average delta ndarrays.
    """
    if not client_deltas:
        raise ValueError("client_deltas list cannot be empty.")

    total_examples = sum(num_examples for _, num_examples in client_deltas)
    if total_examples == 0:
        raise ValueError("Total number of examples across clients is 0.")

    num_tensors = len(client_deltas[0][0])
    # Initialize zero arrays with same shape and float32 dtype
    aggregated_delta = [
        np.zeros_like(arr, dtype=np.float32)
        for arr in client_deltas[0][0]
    ]

    for deltas, num_examples in client_deltas:
        weight = float(num_examples) / float(total_examples)
        for i in range(num_tensors):
            aggregated_delta[i] += (deltas[i] * weight).astype(np.float32)

    return aggregated_delta
