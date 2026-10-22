import numpy as np
import copy

def check_compatibility(base_update, test_update):
    """
    Checks if test_update is compatible with base_update in terms of keys and shapes.
    Raises ValueError if they are incompatible.
    """
    if set(base_update.keys()) != set(test_update.keys()):
        raise ValueError("Incompatible parameter names between updates.")
    
    for key in base_update:
        if base_update[key].shape != test_update[key].shape:
            raise ValueError(f"Incompatible shapes for parameter {key}: expected {base_update[key].shape}, got {test_update[key].shape}")

def generate_canceling_masks(update_schema, num_clients, seed=None):
    """
    Generates a list of canceling masks for a given number of clients based on the update_schema.
    Returns: A list of mask dictionaries. The sum of all mask dictionaries is zero.
    """
    if num_clients < 2:
        raise ValueError("At least 2 clients are required for secure aggregation masking.")
    
    rng = np.random.default_rng(seed)
    masks = [{} for _ in range(num_clients)]
    
    for key, val in update_schema.items():
        shape = val.shape
        dtype = val.dtype
        # Generate random unconstrained masks for the first (N-1) clients
        # For simplicity in prototype, using standard normal distributions scaled up.
        # This acts as our additive mask.
        current_sum = np.zeros(shape, dtype=dtype)
        for i in range(num_clients - 1):
            client_mask = rng.standard_normal(shape).astype(dtype) * 10.0
            masks[i][key] = client_mask
            current_sum += client_mask
            
        # The last client gets the negative sum of all previous masks
        masks[-1][key] = -current_sum

    return masks

def mask_update(update, mask):
    """
    Applies an additive mask to the update.
    Returns a new masked update dictionary.
    """
    check_compatibility(update, mask)
    
    masked = {}
    for key, val in update.items():
        # Using floating-point addition
        masked[key] = val + mask[key]
        
    return masked

def aggregate_masked_updates(masked_updates):
    """
    Aggregates a list of masked updates by summing them.
    Because the masks cancel out, the result should equal the sum of the original updates.
    Returns the aggregated update.
    """
    if not masked_updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
    
    # Use the first update as the schema reference
    base_update = masked_updates[0]
    
    # Validate all other updates against the base
    for update in masked_updates[1:]:
        check_compatibility(base_update, update)
        
    aggregated = {}
    for key in base_update.keys():
        # Start with the first client's parameter
        total = np.zeros_like(base_update[key])
        for update in masked_updates:
            total += update[key]
        aggregated[key] = total
        
    return aggregated
