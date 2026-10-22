import gzip


def compress_update(data):
    return gzip.compress(data)


def decompress_update(data):
    return gzip.decompress(data)
