from privacy.compression import compress_update, decompress_update


def test_compression():
    data = b"This is some sample model update data." * 100

    compressed = compress_update(data)
    restored = decompress_update(compressed)

    assert isinstance(compressed, bytes)
    assert restored == data