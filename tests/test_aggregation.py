import numpy as np
import pytest

from privacy.aggregation import generate_canceling_masks, mask_update, aggregate_masked_updates, check_compatibility
from privacy.dp import apply_differential_privacy
from privacy.serializer import serialize_update, deserialize_update
from privacy.compression import compress_update, decompress_update


def create_schema(dims=(5, 5)):
    return {
        "layer1": np.zeros(dims, dtype=np.float32),
        "layer2": np.zeros(dims, dtype=np.float32)
    }

def test_mask_creation_and_shape_preservation():
    schema = create_schema()
    masks = generate_canceling_masks(schema, num_clients=3, seed=42)
    
    assert len(masks) == 3
    for mask in masks:
        assert mask.keys() == schema.keys()
        for key in mask:
            assert mask[key].shape == schema[key].shape

def test_masking_does_not_mutate_original():
    update = {"param": np.ones((10, 10), dtype=np.float32)}
    original_copy = {"param": np.ones((10, 10), dtype=np.float32)}
    
    mask = {"param": np.ones((10, 10), dtype=np.float32) * 5.0}
    masked = mask_update(update, mask)
    
    np.testing.assert_allclose(update["param"], original_copy["param"])
    np.testing.assert_allclose(masked["param"], np.ones((10, 10), dtype=np.float32) * 6.0)

def test_three_client_mask_cancellation():
    schema = create_schema()
    masks = generate_canceling_masks(schema, num_clients=3, seed=123)
    
    # Calculate mask sum
    for key in schema:
        mask_sum = sum(mask[key] for mask in masks)
        np.testing.assert_allclose(mask_sum, np.zeros_like(schema[key]), atol=1e-5)

def test_aggregation_equals_expected_sum():
    schema = create_schema()
    masks = generate_canceling_masks(schema, num_clients=3, seed=111)
    
    update1 = {"layer1": np.ones((5, 5), dtype=np.float32), "layer2": np.ones((5, 5), dtype=np.float32) * 2}
    update2 = {"layer1": np.ones((5, 5), dtype=np.float32) * 3, "layer2": np.ones((5, 5), dtype=np.float32) * 4}
    update3 = {"layer1": np.ones((5, 5), dtype=np.float32) * 5, "layer2": np.ones((5, 5), dtype=np.float32) * 6}
    
    expected = {
        "layer1": update1["layer1"] + update2["layer1"] + update3["layer1"],
        "layer2": update1["layer2"] + update2["layer2"] + update3["layer2"],
    }
    
    masked1 = mask_update(update1, masks[0])
    masked2 = mask_update(update2, masks[1])
    masked3 = mask_update(update3, masks[2])
    
    aggregated = aggregate_masked_updates([masked1, masked2, masked3])
    
    for key in expected:
        np.testing.assert_allclose(aggregated[key], expected[key], atol=1e-5)

def test_incompatible_names_rejected():
    base = {"layer1": np.zeros((5, 5))}
    test = {"layer2": np.zeros((5, 5))}
    with pytest.raises(ValueError, match="Incompatible parameter names"):
        check_compatibility(base, test)

def test_incompatible_shapes_rejected():
    base = {"layer1": np.zeros((5, 5))}
    test = {"layer1": np.zeros((10, 10))}
    with pytest.raises(ValueError, match="Incompatible shapes"):
        check_compatibility(base, test)

def test_dp_masking_aggregation_integration():
    schema = create_schema()
    
    update1 = {"layer1": np.ones((5, 5), dtype=np.float32) * 1.0, "layer2": np.ones((5, 5), dtype=np.float32) * 1.5}
    update2 = {"layer1": np.ones((5, 5), dtype=np.float32) * 2.0, "layer2": np.ones((5, 5), dtype=np.float32) * 2.5}
    
    # 1. DP
    dp1 = apply_differential_privacy(update1, max_norm=10.0, noise_multiplier=0.1, seed=1)
    dp2 = apply_differential_privacy(update2, max_norm=10.0, noise_multiplier=0.1, seed=2)
    
    # Expected sum of DP updates
    expected = {
        "layer1": dp1["layer1"] + dp2["layer1"],
        "layer2": dp1["layer2"] + dp2["layer2"],
    }
    
    # 2. Masking
    masks = generate_canceling_masks(schema, num_clients=2, seed=999)
    masked1 = mask_update(dp1, masks[0])
    masked2 = mask_update(dp2, masks[1])
    
    # 3. Aggregation
    aggregated = aggregate_masked_updates([masked1, masked2])
    
    for key in expected:
        np.testing.assert_allclose(aggregated[key], expected[key], atol=1e-5)

def test_serialization_compression_compatibility():
    schema = create_schema()
    update = {"layer1": np.ones((5, 5), dtype=np.float32) * 5.0, "layer2": np.ones((5, 5), dtype=np.float32) * 3.0}
    
    # DP
    dp = apply_differential_privacy(update, max_norm=15.0, noise_multiplier=0.1, seed=42)
    
    # Mask
    masks = generate_canceling_masks(schema, num_clients=2, seed=42)
    masked = mask_update(dp, masks[0])
    
    # Serialize -> Compress
    serialized = serialize_update(masked)
    compressed = compress_update(serialized)
    assert isinstance(compressed, bytes)
    
    # Decompress -> Deserialize
    decompressed = decompress_update(compressed)
    restored = deserialize_update(decompressed)
    
    for key in masked:
        np.testing.assert_allclose(masked[key], restored[key])
