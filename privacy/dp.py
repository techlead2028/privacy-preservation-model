import numpy as np

def clip_update(update, max_norm):
    """
    Clips individual layers/parameters of the update based on their L2 norm.
    If the L2 norm of a parameter array exceeds max_norm, it is scaled down.
    Otherwise, it is unchanged.
    Returns a new clipped update dictionary.
    """
    clipped = {}
    for key, val in update.items():
        norm = np.linalg.norm(val)
        if norm > max_norm:
            scaling_factor = max_norm / norm
            clipped[key] = val * scaling_factor
        else:
            clipped[key] = val.copy()
    return clipped

def add_gaussian_noise(update, noise_multiplier, max_norm, seed=None):
    """
    Adds calibrated Gaussian noise to the update.
    The standard deviation of the noise is noise_multiplier * max_norm.
    Returns a new noisy update dictionary.
    """
    noisy = {}
    noise_std = noise_multiplier * max_norm
    rng = np.random.default_rng(seed)
    for key, val in update.items():
        noise = rng.normal(0, noise_std, size=val.shape).astype(val.dtype)
        noisy[key] = val + noise
    return noisy

def apply_differential_privacy(update, max_norm, noise_multiplier, seed=None):
    """
    Applies an academic prototype of a Differential Privacy mechanism to model updates.
    It first clips the update to the specified max_norm, and then adds
    Gaussian noise scaled by noise_multiplier and max_norm.
    
    Returns a protected update dictionary.
    """
    clipped = clip_update(update, max_norm)
    protected = add_gaussian_noise(clipped, noise_multiplier, max_norm, seed=seed)
    return protected
