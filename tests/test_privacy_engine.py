import pytest
import torch
import numpy as np
from privacy.privacy_engine import PrivacyEngine
from privacy.aggregation import generate_canceling_masks, aggregate_masked_updates
from privacy.communication import prepare_payload, decode_payload

@pytest.fixture
def dummy_lora_delta():
    """Generates a dummy 88-tensor LoRA style update."""
    delta = {}
    for i in range(88):
        # random string keys varying to test sorting
        key = f"layer_{i:02d}.lora_A.weight" if i % 2 == 0 else f"layer_{i:02d}.lora_B.weight"
        # 88 tensors, realistic float32 values
        tensor = torch.randn(10, 10, dtype=torch.float32)
        delta[key] = tensor
    return delta

def test_privacy_engine_preserves_shapes_and_values(dummy_lora_delta):
    engine = PrivacyEngine()
    result = engine.process_update(dummy_lora_delta)
    
    assert isinstance(result, list)
    assert len(result) == 88
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        # Check shapes
        assert result[i].shape == dummy_lora_delta[key].shape
        # Check value preservation
        np.testing.assert_allclose(result[i], dummy_lora_delta[key].numpy(), atol=1e-6)
        
def test_privacy_engine_float32(dummy_lora_delta):
    # Change one to float64 to see if it converts back to float32
    key_to_change = list(dummy_lora_delta.keys())[0]
    dummy_lora_delta[key_to_change] = dummy_lora_delta[key_to_change].to(torch.float64)
    
    engine = PrivacyEngine()
    result = engine.process_update(dummy_lora_delta)
    
    for arr in result:
        assert arr.dtype == np.float32

def test_privacy_engine_does_not_mutate_input(dummy_lora_delta):
    engine = PrivacyEngine()
    # take a copy/clone of original
    original_copy = {k: v.clone() for k, v in dummy_lora_delta.items()}
    
    engine.process_update(dummy_lora_delta)
    
    for k in dummy_lora_delta:
        assert torch.equal(dummy_lora_delta[k], original_copy[k])

def test_privacy_engine_deterministic_order():
    delta = {
        "z_layer": torch.ones(2, 2),
        "a_layer": torch.ones(2, 2) * 2,
        "m_layer": torch.ones(2, 2) * 3
    }
    
    engine = PrivacyEngine()
    result = engine.process_update(delta)
    
    # Order should be a_layer, m_layer, z_layer
    assert result[0][0,0] == 2
    assert result[1][0,0] == 3
    assert result[2][0,0] == 1

def test_privacy_engine_dp_disabled(dummy_lora_delta):
    engine = PrivacyEngine()
    config = {"enable_dp": False}
    result = engine.process_update(dummy_lora_delta, config=config)
    
    assert isinstance(result, list)
    assert len(result) == 88
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        assert result[i].shape == dummy_lora_delta[key].shape
        np.testing.assert_allclose(result[i], dummy_lora_delta[key].numpy(), atol=1e-6)

def test_privacy_engine_dp_enabled(dummy_lora_delta):
    engine = PrivacyEngine()
    config = {"enable_dp": True, "max_norm": 10.0, "noise_multiplier": 0.5, "seed": 42}
    result = engine.process_update(dummy_lora_delta, config=config)
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        # 1. output has correct shapes
        assert result[i].shape == dummy_lora_delta[key].shape
        # 2. output remains float32
        assert result[i].dtype == np.float32
        
    # 3. DP processing actually changes the update when noise is enabled
    is_changed = False
    for i, key in enumerate(sorted_keys):
        if not np.allclose(result[i], dummy_lora_delta[key].numpy(), atol=1e-6):
            is_changed = True
            break
    assert is_changed

def test_privacy_engine_deterministic_seed(dummy_lora_delta):
    engine = PrivacyEngine()
    config1 = {"enable_dp": True, "max_norm": 10.0, "noise_multiplier": 0.5, "seed": 42}
    result1 = engine.process_update(dummy_lora_delta, config=config1)
    
    config2 = {"enable_dp": True, "max_norm": 10.0, "noise_multiplier": 0.5, "seed": 42}
    result2 = engine.process_update(dummy_lora_delta, config=config2)
    
    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r1, r2)
        
def test_privacy_engine_different_seed(dummy_lora_delta):
    engine = PrivacyEngine()
    config1 = {"enable_dp": True, "seed": 42, "max_norm": 5.0, "noise_multiplier": 0.5}
    config2 = {"enable_dp": True, "seed": 99, "max_norm": 5.0, "noise_multiplier": 0.5}
    
    result1 = engine.process_update(dummy_lora_delta, config=config1)
    result2 = engine.process_update(dummy_lora_delta, config=config2)
    
    is_different = False
    for r1, r2 in zip(result1, result2):
        if not np.allclose(r1, r2):
            is_different = True
            break
    assert is_different
    
def test_privacy_engine_dp_input_immutability(dummy_lora_delta):
    engine = PrivacyEngine()
    config = {"enable_dp": True, "max_norm": 1.0, "noise_multiplier": 1.0, "seed": 42}
    original_copy = {k: v.clone() for k, v in dummy_lora_delta.items()}
    
    engine.process_update(dummy_lora_delta, config=config)
    
    for k in dummy_lora_delta:
        assert torch.equal(dummy_lora_delta[k], original_copy[k])

def test_privacy_engine_clipping(dummy_lora_delta):
    engine = PrivacyEngine()
    # extremely tiny max_norm, zero noise multiplier means only clipping is applied
    config = {"enable_dp": True, "max_norm": 1e-6, "noise_multiplier": 0.0, "seed": 42}
    result = engine.process_update(dummy_lora_delta, config=config)
    
    for r in result:
        norm = np.linalg.norm(r)
        # using a small tolerance for floating point assertions
        assert norm <= (1e-6 + 1e-10)

def test_privacy_engine_mask_disabled(dummy_lora_delta):
    engine = PrivacyEngine()
    result = engine.process_update(dummy_lora_delta, mask=None)
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        np.testing.assert_allclose(result[i], dummy_lora_delta[key].numpy(), atol=1e-6)
        
def test_privacy_engine_mask_application(dummy_lora_delta):
    engine = PrivacyEngine()
    
    # create a deterministic mask
    mask = {k: np.ones_like(v.numpy()) * 2.0 for k, v in dummy_lora_delta.items()}
    
    result = engine.process_update(dummy_lora_delta, config={"enable_dp": False}, mask=mask)
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        expected = dummy_lora_delta[key].numpy() + 2.0
        np.testing.assert_allclose(result[i], expected, atol=1e-6)

def test_privacy_engine_mask_no_mutation(dummy_lora_delta):
    engine = PrivacyEngine()
    original_copy = {k: v.clone() for k, v in dummy_lora_delta.items()}
    mask = {k: np.ones_like(v.numpy()) for k, v in dummy_lora_delta.items()}
    
    engine.process_update(dummy_lora_delta, mask=mask)
    
    for k in dummy_lora_delta:
        assert torch.equal(dummy_lora_delta[k], original_copy[k])

def test_privacy_engine_mask_shapes_and_float32(dummy_lora_delta):
    engine = PrivacyEngine()
    mask = {k: np.ones_like(v.numpy()) for k, v in dummy_lora_delta.items()}
    result = engine.process_update(dummy_lora_delta, mask=mask)
    
    sorted_keys = sorted(dummy_lora_delta.keys())
    for i, key in enumerate(sorted_keys):
        assert result[i].shape == dummy_lora_delta[key].shape
        assert result[i].dtype == np.float32

def test_privacy_engine_deterministic_masking(dummy_lora_delta):
    engine = PrivacyEngine()
    mask = {k: (np.arange(np.prod(v.shape)).reshape(v.shape) * 0.1).astype(np.float32) for k, v in dummy_lora_delta.items()}
    
    res1 = engine.process_update(dummy_lora_delta, mask=mask)
    res2 = engine.process_update(dummy_lora_delta, mask=mask)
    
    for r1, r2 in zip(res1, res2):
        np.testing.assert_allclose(r1, r2)

def test_privacy_engine_multi_client_cancellation():
    schema = {}
    engine = PrivacyEngine()
    
    # 3 clients
    num_clients = 3
    client_updates = []
    
    for _ in range(num_clients):
        delta = {}
        for i in range(5):
            k = f"weight_{i}"
            delta[k] = torch.randn(5, 5, dtype=torch.float32)
            if k not in schema:
                schema[k] = np.zeros((5, 5), dtype=np.float32)
        client_updates.append(delta)
        
    masks = generate_canceling_masks(schema, num_clients=num_clients, seed=123)
    
    # Process through PrivacyEngine
    engine_results = []
    for i in range(num_clients):
        res = engine.process_update(client_updates[i], mask=masks[i])
        engine_results.append(res)
        
    # Reconstruct dict from engine_results for aggregation
    sorted_keys = sorted(schema.keys())
    dict_results = []
    for res in engine_results:
        recon = {k: res[idx] for idx, k in enumerate(sorted_keys)}
        dict_results.append(recon)
        
    aggregated = aggregate_masked_updates(dict_results)
    
    # Compare to simple sum of originals
    expected_sum = {k: np.zeros((5, 5), dtype=np.float32) for k in sorted_keys}
    for client_idx in range(num_clients):
        for k in sorted_keys:
            expected_sum[k] += client_updates[client_idx][k].numpy()
            
    for k in sorted_keys:
        np.testing.assert_allclose(aggregated[k], expected_sum[k], atol=1e-5)

def test_privacy_engine_dp_plus_masking_ordering(dummy_lora_delta):
    engine = PrivacyEngine()
    config = {"enable_dp": True, "max_norm": 10.0, "noise_multiplier": 0.5, "seed": 42}
    
    # Run DP only
    res_dp_only = engine.process_update(dummy_lora_delta, config=config)
    
    # Make a mask
    mask = {k: np.ones_like(v.numpy()) * 3.0 for k, v in dummy_lora_delta.items()}
    
    # Run DP + Mask
    res_dp_and_mask = engine.process_update(dummy_lora_delta, config=config, mask=mask)
    
    # Verify that the final result = (DP_only_result) + 3.0
    for i in range(len(res_dp_only)):
        np.testing.assert_allclose(res_dp_and_mask[i], res_dp_only[i] + 3.0, atol=1e-5)

def test_privacy_engine_communication_roundtrip(dummy_lora_delta):
    engine = PrivacyEngine()
    
    # 1/5/7/8. Original protected update (List[np.ndarray])
    protected_update = engine.process_update(dummy_lora_delta)
    original_copy = [arr.copy() for arr in protected_update]
    
    # 6. Serialization + Compression
    payload = prepare_payload(protected_update)
    assert isinstance(payload, bytes)
    
    # Decompression + Deserialization
    recovered_update = decode_payload(payload)
    
    assert isinstance(recovered_update, list)
    assert len(recovered_update) == 88
    
    # 2/3/4. Verification
    for orig, rec in zip(protected_update, recovered_update):
        # 4. float32 preserved
        assert rec.dtype == orig.dtype == np.float32
        # 3. Shapes preserved
        assert rec.shape == orig.shape
        # 2. Values preserved
        np.testing.assert_allclose(orig, rec, atol=1e-6)
        
    # 8. No mutation
    for arr_orig, arr_copy in zip(protected_update, original_copy):
        np.testing.assert_array_equal(arr_orig, arr_copy)

def test_privacy_engine_full_e2e_pipeline(dummy_lora_delta):
    """9. DP + masking + serialization + compression end-to-end"""
    engine = PrivacyEngine()
    
    # Setup configuration for DP
    config = {"enable_dp": True, "max_norm": 5.0, "noise_multiplier": 0.1, "seed": 42}
    
    # Setup mask
    mask = {k: np.ones_like(v.numpy()) * 1.5 for k, v in dummy_lora_delta.items()}
    
    # Step 1: Process numerical update (DP + Mask)
    protected_update = engine.process_update(dummy_lora_delta, config=config, mask=mask)
    
    # Step 2: Communication send
    payload = prepare_payload(protected_update)
    
    # Step 3: Communication receive
    recovered_update = decode_payload(payload)
    
    # Verify exact match between protected update and network recovered update
    for orig, rec in zip(protected_update, recovered_update):
        np.testing.assert_allclose(orig, rec, atol=1e-6)
