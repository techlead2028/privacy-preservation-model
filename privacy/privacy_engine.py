import torch
import numpy as np
from typing import Dict, List, Optional, Any
from privacy.dp import apply_differential_privacy
from privacy.aggregation import mask_update

class PrivacyEngine:
    """
    Integration layer between the federated client (Member 2 LoRA) 
    and the privacy modules (Member 3).
    """

    def process_update(
        self, 
        delta: Dict[str, torch.Tensor], 
        num_examples: Optional[int] = None, 
        config: Optional[Dict[str, Any]] = None,
        mask: Optional[Dict[str, np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Validates a Pytorch-based LoRA delta, converts it, applies DP if configured,
        applies a client-specific mask if provided, and returns a Flower-compatible List[np.ndarray].
        
        Note: Mask generation happens outside PrivacyEngine. The current masking mechanism 
        is a prototype additive canceling-mask approach, not production cryptographic secure aggregation.
        """
        if not isinstance(delta, dict):
            raise ValueError(f"delta must be a dict, got {type(delta)}")
            
        config = config or {}

        # 1. Convert to numpy dict
        numpy_delta: Dict[str, np.ndarray] = {}
        for key, tensor in delta.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Value for {key} must be a torch.Tensor, got {type(tensor)}")
            
            arr = tensor.detach().cpu().numpy()
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            numpy_delta[key] = arr
            
        # 2. Apply DP if enabled
        if config.get("enable_dp", False):
            max_norm = config.get("max_norm", 1.0)
            noise_multiplier = config.get("noise_multiplier", 0.1)
            seed = config.get("seed", None)
            
            numpy_delta = apply_differential_privacy(
                numpy_delta, 
                max_norm=max_norm, 
                noise_multiplier=noise_multiplier, 
                seed=seed
            )

        # 3. Apply prototype secure aggregation mask if provided
        if mask is not None:
            numpy_delta = mask_update(numpy_delta, mask)

        # 4. Deterministic parameter ordering (sorted keys)
        flower_update: List[np.ndarray] = []
        sorted_keys = sorted(numpy_delta.keys())
        for key in sorted_keys:
            flower_update.append(numpy_delta[key])
            
        return flower_update
