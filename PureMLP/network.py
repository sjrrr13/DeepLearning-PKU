'''
Pure neural network
'''

import numpy as np
from tqdm import tqdm
import json
import math
from utils import *


BGD = 0
SGD = 1
MBGD = 2
MBSGD = 3

np.random.seed(2024)

class Layer:
    def __init__(self):
        pass
        
    def forward(self, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`forward` must be overridden")
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`backward` must be overridden")
    
    def learn(self, learn_rate: float):
        raise NotImplementedError("`learn` must be overridden")


class FCLayer(Layer):
    def __init__(
        self, 
        in_size: int, 
        out_size: int,
        l1: int=0,
        l2: int=0,
    ):
        super().__init__()
        # Generate weight and bias from normal distribution
        scale = np.sqrt(2. / (in_size + out_size))
        self.weight = np.random.normal(loc=0, scale=scale, size=(in_size, out_size))
        self.bias = np.random.normal(loc=0, scale=scale, size=(1, out_size))
        
        # Generate weight and bias randomly
        # scale = 0.01
        # self.weight = np.random.randn(in_size, out_size) * scale
        # self.bias = np.random.randn(1, out_size) * scale

        self.l1 = l1
        self.l2 = l2        
        self.d_weight, self.d_bias = None, None

    def forward(self, v: np.ndarray) -> np.ndarray:
        self.input = v
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.d_weight = np.dot(self.input.T, grad)
        self.d_bias = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weight.T)

    def learn(self, learn_rate: float):
        if self.l1 != 0:
            # L1 regularization
            self.weight -= learn_rate * (self.d_weight + self.l1 * np.sign(self.weight))
        elif self.l2 != 0:
            # L2 regularization
            self.weight -= learn_rate * (self.d_weight + self.l2 * self.weight)
        else:
            # 不进行正则化
            self.weight -= learn_rate * self.d_weight
        
        self.bias  -= learn_rate * self.d_bias
        

class SigmoidLayer(Layer):
    def __init__( self):
        super().__init__()

    def forward(self, v: np.ndarray) -> np.ndarray:
        # Optimizing sigmoid to avoid overflow
        self.output = np.zeros_like(v)
        mask = (v >= 0)
        self.output[mask] = 1.0 / (1.0 + np.exp(-v[mask]))
        self.output[~mask] = np.exp(v[~mask]) / (1 + np.exp(v[~mask]))
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.output * (1 - self.output)

    def learn(self, learn_rate: float):
        pass


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        # Optimizing softmax to avoid overflow
        exp_v = np.exp(v - np.max(v, axis=1, keepdims=True))
        self.output = exp_v / np.sum(exp_v, axis=1, keepdims=True)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # This function do nothing, as the gradient of cross entropy and
        # softmax had been computed together in `cross_entropy`
        return grad

    def learn(self, learn_rate: float):
        pass


class MLPNet:
    def __init__(
        self, 
        network, 
        learn_rate: float,
        category: int,
        l1: int,
        l2: int
    ):
        self.layers = []
        self.learn_rate = learn_rate
        self.category = category
        assert network[-1] == category, "Output of the last hidden layer must match categories"

        for i in range(len(network) - 2):
            self.layers.append(FCLayer(network[i], network[i + 1], l1=l1, l2=l2))
            self.layers.append(SigmoidLayer())

        self.layers.append(FCLayer(network[-2], network[-1], l1=l1, l2=l2))
        self.layers.append(SoftmaxLayer())

    def forward(self, v: np.ndarray) -> np.ndarray:
        hidden = v
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return hidden

    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def learn(self):
        for layer in self.layers:
            layer.learn(self.learn_rate)

    def classify(self, input: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(input), axis=1)

    # Print structure of the network, only for debugging
    def struct(self):
        for layer in self.layers:
            if hasattr(layer, "in_size") and hasattr(layer, "out_size"):
                print(f"{type(layer).__name__}: {layer.in_size} -> {layer.out_size}")
            else:
                print(f"{type(layer).__name__}")


def train_sgd(
    network, 
    train_data: np.ndarray, 
    train_label: np.ndarray,
    test_data: np.ndarray,
    test_label: np.ndarray
):
    idx = np.random.randint(0, len(train_data))
    train = np.expand_dims(train_data[idx], axis=0)
    label = np.expand_dims(train_label[idx], axis=0)
    output = network.forward(train)
    loss, grad = cross_entropy(output, label)

    network.backward(grad)
    network.learn()

    accuracy = evaluate(network.classify(test_data), test_label)
    return {"loss": loss, "accuracy": accuracy}


def train_bgd(
    network, 
    train_data: np.ndarray, 
    train_label: np.ndarray,
    test_data: np.ndarray,
    test_label: np.ndarray
):
    output = network.forward(train_data)
    loss, grad = cross_entropy(output, train_label)

    network.backward(grad)
    network.learn()

    accuracy = evaluate(network.classify(test_data), test_label)
    return {"loss": loss, "accuracy": accuracy}


def single_train(
    network, 
    train_data: np.ndarray, 
    train_label: np.ndarray,
):
    output = network.forward(train_data)
    loss, grad = cross_entropy(output, train_label)
    network.backward(grad)
    network.learn()
    return loss


def train(
    network, 
    train_data: np.ndarray, 
    train_label: np.ndarray,
    test_data: np.ndarray,
    test_label: np.ndarray,
    type: int=BGD,
    epoch: int=100,
    bs: int=10,
    gamma: float=1.0,
    step_size: int=0
):
    log = []
    lr = network.learn_rate
    _lr = network.learn_rate
    
    for e in tqdm(range(epoch)):
        if type == BGD:
            train = train_data
            label = train_label
        elif type == SGD:
            idx = np.random.randint(0, len(train_data))
            train = np.expand_dims(train_data[idx], axis=0)
            label = np.expand_dims(train_label[idx], axis=0)
        elif type == MBGD:
            b_train, b_label = batchfy_data(train_data, train_label, bs=bs)
            idx = np.random.randint(0, len(b_train))    # Choose a mini batch
            train = b_train[idx]
            label = b_label[idx]
        elif type == MBSGD:
            train, label = [], []
            for _ in range(bs): # Construct a mini batch
                idx = np.random.randint(0, len(train_data))
                train.append(train_data[idx])
                label.append(train_label[idx])
            train = np.array(train)
            label = np.array(label)
        else:
            print("Unknown training algorithm, use BGD as default.")
            train = train_data
            label = train_label
        
        loss = single_train(network, train, label)
        accuracy = evaluate(network.classify(test_data), test_label)
        log.append({"loss": loss, "accuracy": accuracy})
        
        if step_size != 0:  # lr decay method1
            if (e + 1) % step_size == 0:
                lr *= gamma
                network.learn_rate = lr
        elif gamma != 1:    # lr decay method2
            lr = math.pow(gamma, e) * _lr
            network.learn_rate = lr

    return log


def main():
    filename = "seed_1000_2"
    network = MLPNet(
        network=[28*28, 512, 16*16, 10],
        learn_rate=1e-6,
        category=CATEGORY,
        l1=0,
        l2=0
    )
    train_img, train_label, test_img, test_label = load_MNIST()

    log = train(network, train_img, train_label, test_img, test_label,
          type=BGD, 
          epoch=1000,
          step_size=0,
          gamma=1.0)

    with open(f"./Result/{filename}.jsonl", "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")

    draw_plt(filename, title="Add a layer")


if __name__ == "__main__":
    main()
    