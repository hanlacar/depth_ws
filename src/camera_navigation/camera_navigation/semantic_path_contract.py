"""Decode the lossless typed semantic-path RLE contract."""
import numpy as np


def decode_binary_rle(values, height, width):
    encoded = np.asarray(values, dtype=np.uint32)
    if encoded.size % 2:
        raise ValueError("RLE must contain start/end pairs")
    size = int(height)*int(width)
    output = np.zeros(size, np.uint8)
    for start, end in encoded.reshape(-1, 2):
        if int(start) > int(end) or int(end) > size:
            raise ValueError("RLE range outside image")
        output[int(start):int(end)] = 255
    return output.reshape(int(height), int(width))
