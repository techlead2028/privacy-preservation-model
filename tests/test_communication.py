import numpy as np

from privacy.communication import prepare_client_payload, decode_client_payload, prepare_payload, decode_payload
from privacy.aggregation import generate_canceling_masks, mask_update
from privacy.dp import apply_differential_privacy

def create_lora_update(size_multiplier=1):
    return {
        "layer1": np.ones((5 * size_multiplier, 5 * size_multiplier), dtype=np.float32),
        "layer2": np.ones((5 * size_multiplier, 5 * size_multiplier), dtype=np.float32) * 2.0
    }

def test_communication_pipeline_no_protections():
    update = create_lora_update()
    
    payload = prepare_client_payload(update)
    assert isinstance(payload, bytes)
    
    restored = decode_client_payload(payload)
    
    assert restored.keys() == update.keys()
    for key in update:
        np.testing.assert_allclose(update[key], restored[key])

def test_communication_pipeline_with_dp():
    update = create_lora_update()
    dp_args = {"max_norm": 5.0, "noise_multiplier": 0.1, "seed": 42}
    
    payload = prepare_client_payload(update, dp_args=dp_args)
    assert isinstance(payload, bytes)
    
    restored = decode_client_payload(payload)
    
    # Should not strictly equal the original due to noise
    for key in update:
        assert not np.array_equal(update[key], restored[key])

def test_communication_pipeline_with_mask():
    update = create_lora_update()
    masks = generate_canceling_masks(update, num_clients=2, seed=123)
    
    payload = prepare_client_payload(update, mask=masks[0])
    assert isinstance(payload, bytes)
    
    restored = decode_client_payload(payload)
    
    # Should track the masked value appropriately
    expected_masked = update["layer1"] + masks[0]["layer1"]
    np.testing.assert_allclose(expected_masked, restored["layer1"], atol=1e-5)

def test_full_communication_pipeline():
    update = create_lora_update()
    dp_args = {"max_norm": 10.0, "noise_multiplier": 0.1, "seed": 99}
    masks = generate_canceling_masks(update, num_clients=2, seed=111)
    
    payload = prepare_client_payload(update, mask=masks[0], dp_args=dp_args)
    assert isinstance(payload, bytes)
    
    restored = decode_client_payload(payload)
    
    assert restored.keys() == update.keys()
    for key in restored:
        assert restored[key].shape == update[key].shape

# ========================================================
# CLEAN API TESTS
# ========================================================

def test_clean_prepare_and_decode_payload():
    update = create_lora_update()
    
    # Pack clean
    payload = prepare_payload(update)
    assert isinstance(payload, bytes)
    
    # Unpack clean
    restored = decode_payload(payload)
    
    assert update.keys() == restored.keys()
    for key in update:
        np.testing.assert_allclose(update[key], restored[key])

def test_clean_architectural_flow_orchestration():
    # DP -> Masking -> prepare_payload -> decode_payload
    update = create_lora_update()
    
    # 1. DP independently applied
    dp_update = apply_differential_privacy(update, max_norm=5.0, noise_multiplier=0.1, seed=42)
    
    # 2. Masking independently applied
    masks = generate_canceling_masks(update, num_clients=2, seed=123)
    masked_update = mask_update(dp_update, masks[0])
    
    # 3. Payload serialization directly
    payload = prepare_payload(masked_update)
    assert isinstance(payload, bytes)
    
    # 4. Decoding backwards 
    restored = decode_payload(payload)
    
    assert masked_update.keys() == restored.keys()
    for key in masked_update:
        np.testing.assert_allclose(masked_update[key], restored[key])
