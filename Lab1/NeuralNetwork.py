'''
Pure neural network
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm

TRAIN_DATA = "MNIST/train-images-idx3-ubyte"
TRAIN_LABEL = "MNIST/train-labels-idx1-ubyte"
TEST_DATA = "MNIST/t10k-images-idx3-ubyte"
TEST_LABEL = "MNIST/t10k-labels-idx1-ubyte"


class Layer:
    def __init__(
        self,
        in_size: int,
        out_size: int
    ):
        self.in_size, self.out_size = in_size, out_size
        self.input, self.output = None, None
        
    def forward(self, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`forward` must be overridden")
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`backward` must be overridden")
    
    def learn(self, learn_rate: float):
        raise NotImplementedError("`learn` must be overridden")


class HiddenLayer (Layer):
    def __init__(
        self, 
        in_size: int, 
        out_size: int
    ):
        super().__init__(in_size, out_size)
        
        # Generate weight and bias for normal distribution
        self.weight = np.random.normal(loc=0, scale=1, size=(in_size, out_size))
        self.bias = np.random.normal(loc=0, scale=1, size=(1, out_size))
        self.d_weight, self.d_bias = None, None

    def forward(self, v: np.ndarray) -> np.ndarray:
        self.input = v
        self.output = self.bias + np.dot(self.input, self.weight)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.d_weight = np.matmul(self.input.T, grad)   # Gradient of Sigmoid function
        self.d_bias = np.average(grad, 0)
        return np.matmul(grad, self.weight.T)

    def learn(self, learn_rate: float):
        self.weight -= learn_rate * self.d_weight
        self.bias -= learn_rate * self.d_bias
        

class SigmoidLayer (Layer):
    def __init__(
        self, 
        in_size: int, 
        out_size: int
    ):
        super().__init__(in_size, out_size)

    def forward(self, v: int) -> np.ndarray:
        exp_v = np.exp(v - np.max(v))   # Optimized sigmoid
        self.output = exp_v / np.sum(exp_v)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.output * (1 - self.output) * grad

    def learn(self, learn_rate: float):
        pass


class SoftmaxLayer (Layer):
    def __init__(
        self, 
        in_size: int,
        out_size:int
    ):
        super().__init__(in_size, out_size)

    def forward(self, v: np.ndarray) -> np.ndarray:
        e_v = np.exp(v)
        self.output = e_v / np.sum(e_v)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad

    def learn(self, learn_rate: float):
        pass


class Network:
    def __init__(
        self, 
        network, 
        learn_rate: float=0.01,
        category: int=10
    ):
        self.layers = []
        self.learn_rate = learn_rate
        self.category = category
        
        assert network[-1] == category, "Output of the last hidden layer must match categories"
        for i in range(len(network) - 1):
            self.layers.append(HiddenLayer(network[i], network[i + 1]))
            self.layers.append(SigmoidLayer(network[i + 1], network[i + 1]))

        # Add a softmax layer for classification task
        self.layers.append(SoftmaxLayer(self.category, self.category))

    def forward(self, v: np.ndarray) -> np.ndarray:
        hidden = v                      # [bs, in_size]
        for layer in self.layers:
            hidden = layer.forward(hidden)   # [bs, out_size]
        return hidden

    # Backward Propagation
    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def learn(self):
        for layer in self.layers:
            layer.learn(self.learn_rate)

    def classify(self, input: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(input))   # [bs, category]

    def train_bgd(
        self, 
        data: np.ndarray, # [num_data, row*col + 1]
        category: int=10,
        epoch: int=10,
    ):
        num_data = len(data)
        for i in range(epoch):
            loss = 0
            grad = np.zeros(category)
            batch_data = batchfy_data(data, bs=1)
            for d in tqdm(batch_data, desc=f"epoch {i}"):
                output = self.forward(d[:, :-1])            # [bs, category]
                truth = np.zeros((d.shape[0], category))
                truth[np.arange(d.shape[0]), d[:, -1]] = 1  # [bs, category]
                tmp_loss, tmp_grad = cross_entropy(output, truth, category)
                loss += tmp_loss
                grad += tmp_grad
            loss /= num_data
            grad /= num_data
            self.backward(grad)
            self.learn()
            print(f"loss in epoch {i}: {loss}")
            
    def train_sgd():
        pass

    # Print structure of the network, only for debugging
    def struct(self):
        for layer in self.layers:
            print(f"{type(layer).__name__}: {layer.in_size} -> {layer.out_size}")
    
    
if __name__ == "__main__":
    network = Network(
        network=[28*28, 20*20, 10*10, 10],
        learn_rate=0.9,
        category=10
    )
    
    data_shape, data = read_idx(TRAIN_DATA)
    label_shape, label = read_idx(TRAIN_LABEL)
    data = merge_data_label(data, label)

    network.train_bgd(data)

    # batch_data = np.array([[1, 2, 3, 0], [4, 5, 6, 1]])
    # batch_img = batch_data[:, :-1]
    # batch_label = batch_data[:, -1]
    # # print(batch_img)
    # # print(batch_label)
    # grad = np.zeros(len(batch_data))
    # grad += batch_data
    # grad += batch_data
    # print(grad)