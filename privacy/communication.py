from privacy.serializer import serialize_update, deserialize_update
from privacy.compression import compress_update, decompress_update
from privacy.dp import apply_differential_privacy
from privacy.aggregation import mask_update


def prepare_payload(update):
    """
    Clean communication interface to convert an already-processed model update 
    into a compressed bytes payload for transmission.
    
    Sequentially applies:
    1. Serialization
    2. Compression
    
    Returns: Transmittable compressed bytes payload.
    """
    # 1. Serialization
    serialized = serialize_update(update)
    
    # 2. Compression
    compressed = compress_update(serialized)
    
    return compressed


def decode_payload(payload):
    """
    Clean communication interface to unpack a received bytes payload.
    
    Sequentially applies:
    1. Decompression
    2. Deserialization
    
    Returns: Recovered update dictionary.
    """
    # 1. Decompression
    decompressed = decompress_update(payload)
    
    # 2. Deserialization
    restored = deserialize_update(decompressed)
    
    return restored


# =====================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =====================================================================

def prepare_client_payload(update, mask=None, dp_args=None):
    """
    [Legacy Wrapper] Client-side communication interface mapping full transformations.
    Please use independent transformations (dp.py, aggregation.py) and 
    pass the final processed update to prepare_payload() instead.
    """
    processed = update
    
    if dp_args is not None:
        processed = apply_differential_privacy(
            processed,
            max_norm=dp_args.get("max_norm", 1.0),
            noise_multiplier=dp_args.get("noise_multiplier", 0.1),
            seed=dp_args.get("seed", None)
        )
        
    if mask is not None:
        processed = mask_update(processed, mask)
        
    return prepare_payload(processed)


def decode_client_payload(payload):
    """
    [Legacy Wrapper] Server-side communication interface unpacker.
    Maps directly to decode_payload.
    """
    return decode_payload(payload)
