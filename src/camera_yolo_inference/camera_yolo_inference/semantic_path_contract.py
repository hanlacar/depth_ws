"""Compact, lossless binary-mask transport for one inference frame."""
import numpy as np


def encode_binary_rle(mask):
    flat = (np.asarray(mask).reshape(-1) > 0).astype(np.int8)
    padded = np.pad(flat, (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return changes.astype(np.uint32)


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


def encoded_payload_bytes(*arrays):
    return sum(np.asarray(array, dtype=np.uint32).nbytes for array in arrays)
