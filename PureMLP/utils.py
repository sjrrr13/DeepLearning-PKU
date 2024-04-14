'''
Util functions
'''

import numpy as np
import struct
import matplotlib.pyplot as plt
from typing import Tuple
import json


TRAIN_DATA = "MNIST/train-images-idx3-ubyte"
TRAIN_LABEL = "MNIST/train-labels-idx1-ubyte"
TEST_DATA = "MNIST/t10k-images-idx3-ubyte"
TEST_LABEL = "MNIST/t10k-labels-idx1-ubyte"

CATEGORY = 10


def read_idx(filename):
    with open(filename) as f:
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            assert zero == 0, "Error magic number"
            assert data_type == 0x08, "Data type should equal to `0x08`"
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape) 


def load_MNIST():
    train_img = read_idx(TRAIN_DATA)
    train_label = read_idx(TRAIN_LABEL)
    test_img = read_idx(TEST_DATA)
    test_label = read_idx(TEST_LABEL)

    # Preprocess
    train_img = train_img.reshape(train_img.shape[0], -1)   # [60000, 784]
    test_img = test_img.reshape(test_img.shape[0], -1)  # [60000, 784]
    train_img = batch_norm(train_img)
    test_img = batch_norm(test_img)

    tmp = np.zeros((train_label.shape[0], CATEGORY))    # one-hot train label
    for i, value in enumerate(train_label):
        tmp[i, value] = 1
    train_label = tmp   # [60000, 10]
    
    return train_img, train_label, test_img, test_label


def batch_norm(data: np.ndarray):
    mean = np.mean(data, axis=0, keepdims=True)
    variance = np.var(data, axis=0, keepdims=True)
    x_centered = data - mean
    stddev_inv = 1.0 / np.sqrt(variance + 1e-8)
    return x_centered * stddev_inv


def cross_entropy(
    output: np.ndarray, 
    truth: np.ndarray,
) -> Tuple[float, np.ndarray]:
    # Add epsilon to avoid log(0)
    loss = -np.mean(np.sum(truth * np.log(output + 1e-15), axis=1))
    # Gradient of cross entropy and softmax
    grad = output - truth
    return loss, grad


def evaluate(
    result: np.ndarray,
    label: np.ndarray
) -> float:
    assert len(result) == len(label)
    return np.sum(np.equal(result, label)) / len(result)


def batchfy_data(data: np.ndarray, label: np.ndarray, bs: int=1) -> list:
    assert len(data) == len(label)
    batch_data, batch_label = [], []
    batch_start = 0
    while batch_start < len(data):
        batch_data += [data[batch_start : min(batch_start + bs, len(data))]]
        batch_label += [label[batch_start : min(batch_start + bs, len(data))]]
        batch_start += bs
    return batch_data, batch_label


def draw_plt(filename, title):
    loss = []
    acc = []
    with open(f"./Output/{filename}.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            loss.append(data["loss"])
            acc.append(data["accuracy"])

    plt.plot(loss)
    plt.title(f"Loss of {title}")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig(f"./Figs/{filename}-loss.png")
    
    plt.close()
    plt.plot(acc)
    plt.title(f"Accuracy of {title}")
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"./Figs/{filename}-acc.png")
    