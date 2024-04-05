'''
Util functions
'''

import numpy as np
import struct

def read_idx(filename):
    with open(filename) as f:
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            assert zero == 0, "Error magic number"
            assert data_type == 0x08, "Data type should equal to `0x08`"
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return shape, np.frombuffer(f.read(), dtype=np.uint8).reshape(shape) 

def cross_entropy(
    output: np.ndarray, 
    truth: np.ndarray,
    category: int
) -> tuple[float, np.ndarray]:
    loss = -1 * np.mean(
        np.sum(truth * np.log(output), axis=1)
        )
    # Gradient of cross entropy and softmax
    grad = np.sum(output - truth, axis=0) / truth.shape[0]
    return loss, grad

def evaluate(
    result: np.ndarray,
    label: np.ndarray
) -> float:
    return np.sum(np.equal(result, label)) / len(result)

def merge_data_label(img: np.ndarray, label: np.ndarray): 
    assert len(img.shape) == 3
    assert len(label.shape) == 1
    flat_img = img.reshape(img.shape[0], -1)    # [num_data, row*col]
    exp_label = np.expand_dims(label, axis=1)   # [num_data, 1]
    return np.concatenate((flat_img, exp_label), axis=1) # [num_data, row*col + 1]

def batchfy_data(data: np.ndarray, bs: int=1) -> list:
    batch_data = []
    batch_start = 0
    while batch_start < len(data):
        batch_data += [data[batch_start : min(batch_start + bs, len(data))]]
        batch_start += bs
    return batch_data
