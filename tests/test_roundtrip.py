import numpy as np

from privacy.serializer import serialize_update, deserialize_update
from privacy.compression import compress_update, decompress_update


def create_lora_update(size_modifier=1):
    """Simulates a realistic LoRA-style update dictionary."""
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(8 * size_modifier, 128 * size_modifier).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(128 * size_modifier, 8 * size_modifier).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": np.random.randn(8 * size_modifier, 128 * size_modifier).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": np.random.randn(128 * size_modifier, 8 * size_modifier).astype(np.float32)
    }

def test_roundtrip_lora_update():
    # 1. Real LoRA-style update
    original_update = create_lora_update()
    
    # 2. Serialize
    serialized = serialize_update(original_update)
    assert isinstance(serialized, bytes)
    
    # 3. Compress
    compressed = compress_update(serialized)
    assert isinstance(compressed, bytes)
    
    # 4. Decompress
    decompressed = decompress_update(compressed)
    assert isinstance(decompressed, bytes)
    assert decompressed == serialized
    
    # 5. Deserialize
    restored_update = deserialize_update(decompressed)
    
    # Verify all properties
    assert original_update.keys() == restored_update.keys()
    for key in original_update:
        orig = original_update[key]
        rest = restored_update[key]
        
        assert orig.shape == rest.shape
        assert orig.dtype == rest.dtype
        np.testing.assert_allclose(orig, rest)
