import numpy as np

from privacy.dp import clip_update, add_gaussian_noise, apply_differential_privacy
from privacy.serializer import serialize_update, deserialize_update
from privacy.compression import compress_update, decompress_update


def test_clipping_below_threshold():
    # norm = sqrt(100*0.01) = 1.0 (which is <= max_norm=2.0)
    update = {"param": np.ones((10, 10)) * 0.1}
    max_norm = 2.0
    
    clipped = clip_update(update, max_norm)
    
    assert "param" in clipped
    assert clipped["param"].shape == update["param"].shape
    np.testing.assert_allclose(clipped["param"], update["param"])

def test_clipping_above_threshold():
    # norm = sqrt(100 * 100) = 100.0
    update = {"param": np.ones((10, 10)) * 10.0}
    max_norm = 10.0
    
    clipped = clip_update(update, max_norm)
    
    norm = np.linalg.norm(clipped["param"])
    np.testing.assert_allclose(norm, max_norm)
    
def test_clipping_zero_vector():
    update = {"param": np.zeros((5, 5))}
    clipped = clip_update(update, max_norm=1.0)
    np.testing.assert_allclose(clipped["param"], update["param"])

def test_gaussian_noise():
    update = {"param": np.zeros((100, 100))}
    noisy = add_gaussian_noise(update, noise_multiplier=1.0, max_norm=1.0, seed=42)
    # The noise should make the array non-zero
    assert not np.allclose(noisy["param"], update["param"])
    assert noisy["param"].shape == update["param"].shape
    
def test_deterministic_noise():
    update = {"param": np.ones((10, 10))}
    noisy_1 = add_gaussian_noise(update, noise_multiplier=0.5, max_norm=2.0, seed=123)
    noisy_2 = add_gaussian_noise(update, noise_multiplier=0.5, max_norm=2.0, seed=123)
    np.testing.assert_allclose(noisy_1["param"], noisy_2["param"])

def test_no_mutation_of_original():
    update = {"param": np.ones((5, 5)) * 10.0}
    original_copy = {"param": np.ones((5, 5)) * 10.0}
    
    apply_differential_privacy(update, max_norm=1.0, noise_multiplier=1.0)
    np.testing.assert_allclose(update["param"], original_copy["param"])

def test_api_pipeline():
    update = {"param": np.ones((10, 10)) * 2.0} # Norm is 20.0
    protected = apply_differential_privacy(update, max_norm=5.0, noise_multiplier=0.1, seed=42)
    
    assert "param" in protected
    assert protected["param"].shape == update["param"].shape
    
def test_integration_with_serialization_and_compression():
    update = {
        "layer1": np.ones((10, 10)) * 5.0,
        "layer2": np.ones((5, 5)) * -2.0
    }
    
    # Apply DP
    protected = apply_differential_privacy(update, max_norm=10.0, noise_multiplier=0.5, seed=111)
    
    # Serialize
    serialized = serialize_update(protected)
    assert isinstance(serialized, bytes)
    
    # Compress
    compressed = compress_update(serialized)
    assert isinstance(compressed, bytes)
    
    # Decompress
    decompressed = decompress_update(compressed)
    
    # Deserialize
    restored = deserialize_update(decompressed)
    
    assert restored.keys() == protected.keys()
    for key in protected:
        assert restored[key].dtype == protected[key].dtype
        assert restored[key].shape == protected[key].shape
        np.testing.assert_allclose(restored[key], protected[key])
