import numpy as np

from privacy.serializer import serialize_update, deserialize_update


def test_serialization():
    # Simulated LoRA adapter update
    update = {
        "layer1.lora_A": np.random.randn(4, 8),
        "layer1.lora_B": np.random.randn(8, 4),
        "layer2.lora_A": np.random.randn(4, 16),
        "layer2.lora_B": np.random.randn(16, 4)
    }

    # Serialize
    serialized = serialize_update(update)

    # Deserialize
    restored = deserialize_update(serialized)

    # Verify serialized data is bytes
    assert isinstance(serialized, bytes)

    # Verify all parameters were restored correctly
    assert update.keys() == restored.keys()

    for key in update:
        assert np.array_equal(update[key], restored[key])