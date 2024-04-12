'''
Pure neural network
'''

import numpy as np
from utils import *
from tqdm import tqdm
import sys
import json

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
        self.name = f"Hidden-{in_size}->{out_size}"
        
        # Generate weight and bias for normal distribution
        self.weight = np.random.normal(loc=0, scale=1, size=(in_size, out_size))
        self.bias = np.random.normal(loc=0, scale=1, size=(1, out_size))
        self.d_weight, self.d_bias = None, None

    def forward(self, v: np.ndarray) -> np.ndarray:
        self.input = v
        self.output = self.bias + np.dot(self.input, self.weight)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # print(f"{self.name}: {self.input.T.shape}")
        # print(f"{self.name}: {grad.shape}")
        self.d_weight = np.matmul(self.input.T, grad)
        # print(f"{self.name}: {self.d_weight}")
        self.d_bias = np.average(grad, axis=0)
        # print(f"{self.name}: {self.d_bias}")
        # print(self.weight.T)
        # print(np.matmul(grad, self.weight.T))
        # exit()
        # print(grad.shape)
        # print(self.weight.T.shape)
        # print(np.matmul(grad, self.weight.T).shape)
        return np.matmul(grad, self.weight.T)

    def learn(self, learn_rate: float):
        # print(f"{self.name}: {self.d_weight}, sum is {np.sum(self.d_weight)}\n"
            #   f"{self.d_bias}, sum is {np.sum(self.d_bias)}\n")
        self.weight -= learn_rate * self.d_weight
        self.bias -= learn_rate * self.d_bias
        

class SigmoidLayer (Layer):
    def __init__(
        self, 
        in_size: int, 
        out_size: int
    ):
        super().__init__(in_size, out_size)
        self.name = f"Sigmoid-{in_size}->{out_size}"

    def forward(self, v: int) -> np.ndarray:
        # self.output = 1.0 / (1.0 + np.exp(-v))
        self.output = v.copy()      # 对sigmoid函数优化，避免出现极大的数据溢出
        self.output[v >= 0] = 1.0 / (1 + np.exp(-v[v >= 0]))
        self.output[v < 0] = np.exp(v[v < 0]) / (1 + np.exp(v[v < 0]))
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # print(self.output)
        # print((1 - self.output))
        # print(self.output * (1 - self.output))
        # print(self.output * (1 - self.output) * grad)
        # exit()
        return self.output * (1 - self.output) * grad

    def learn(self, learn_rate: float):
        pass


class ReluLayer (Layer):
    def __init__(
        self, 
        in_size: int, 
        out_size: int
    ):
        super().__init__(in_size, out_size)
        self.name = f"Relu-{in_size}->{out_size}"

    def forward(self, v: int) -> np.ndarray:
        self.input = v
        self.output = np.maximum(0, v)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        tmp = np.zeros(self.input.shape)
        tmp[self.input>0] = 1
        tmp[self.input==0] = 0.5
        # print(tmp)
        # print(grad)
        return tmp * grad

    def learn(self, learn_rate: float):
        pass


class SoftmaxLayer (Layer):
    def __init__(
        self, 
        in_size: int,
        out_size:int
    ):
        super().__init__(in_size, out_size)
        self.name = f"Softmax-{in_size}->{out_size}"

    def forward(self, v: np.ndarray) -> np.ndarray:
        exp_v = np.exp(v - np.max(v))   # Optimized softmax
        self.output = exp_v / np.sum(exp_v)
        return self.output

    # This function do nothing, as the gradient of cross entropy and
    # softmax had been computed in cross_entropy function
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.expand_dims(grad, axis=0)

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
        for i in range(len(network) - 2):
            self.layers.append(HiddenLayer(network[i], network[i + 1]))
            # self.layers.append(SigmoidLayer(network[i + 1], network[i + 1]))
            self.layers.append(ReluLayer(network[i + 1], network[i + 1]))

        self.layers.append(HiddenLayer(network[-2], network[-1]))
        # Add a softmax layer for classification task
        self.layers.append(SoftmaxLayer(self.category, self.category))

    def forward(self, v: np.ndarray) -> np.ndarray:
        hidden = v / np.max(v)  # [bs, in_size]
        # print(hidden)
        for layer in self.layers:
            hidden = layer.forward(hidden)   # [bs, out_size]
            # print(f"{layer.name}: {hidden}")
        # exit()
        return hidden

    # Backward Propagation
    def backward(self, grad: np.ndarray):
        # print(f"after cross entropy: {grad}")
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            # print(f"after {layer.name}: {grad}")
        # exit()
    
    def learn(self):
        for layer in self.layers:
            layer.learn(self.learn_rate)
        # exit()

    def classify(self, input: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(input), axis=1)   # [bs, category]

    def train_bgd(
        self, 
        train_data: np.ndarray, # [num_data, row*col + 1]
        test_data: np.ndarray,
        test_label: np.ndarray,
        category: int=10,
        epoch: int=10,
    ):
        num_data = len(train_data)
        loss_log = [sys.maxsize]
        accuracy_log = [0]
        for i in range(epoch):
            accuracy = evaluate(self.classify(test_data), test_label)
            if not accuracy > accuracy_log[-1]:
                print("accuracy do not increase, break")
                break
            else:
                accuracy_log.append(accuracy)

            loss = 0
            grad = np.zeros(category)
            for d in data:
                input_d = d[:-1].reshape(1, -1)
                output = self.forward(input_d)
                truth = np.zeros((1, category))
                truth[0][d[-1]] = 1
                tmp_loss, tmp_grad = cross_entropy(output, truth, category)
                loss += tmp_loss
                grad += tmp_grad
            loss /= num_data
            grad /= num_data
            print(grad.shape)
            print(f"epoch {i}: accuracy: {accuracy}, loss: {loss}")
            if not loss < loss_log[-1]:
                print("loss do not decrease, break")
                break
            else: 
                loss_log.append(loss)
                self.backward(grad)
                self.learn()
            sys.stdout.flush()
        
        return loss_log
    
    def train_bgd_new(
        self, 
        train_data: np.ndarray, # [num_data, row*col + 1]
        category: int=10,
    ):
        num_data = len(train_data)

        loss = 0
        grad = np.zeros(category)
        for d in data:
            input_d = d[:-1].reshape(1, -1)
            output = self.forward(input_d)
            truth = np.zeros((1, category))
            truth[0][d[-1]] = 1
            tmp_loss, tmp_grad = cross_entropy(output, truth)
            loss += tmp_loss
            grad += tmp_grad
        
        loss /= num_data
        grad /= num_data
        # print(loss)
        # print(grad)
        # exit()
        self.backward(grad)
        self.learn()
        
        result = {"loss": loss}
        return result
            
    def train_sgd():
        pass

    # Print structure of the network, only for debugging
    def struct(self):
        for layer in self.layers:
            print(f"{type(layer).__name__}: {layer.in_size} -> {layer.out_size}")
    
    
if __name__ == "__main__":
    network = Network(
        network=[28*28, 16*16, 10],
        learn_rate=0.01,
        category=10
    )
    
    data_shape, data = read_idx(TRAIN_DATA)
    label_shape, label = read_idx(TRAIN_LABEL)
    data = merge_data_label(data, label)

    test_shape, test = read_idx(TEST_DATA)
    test_label_shape, test_label = read_idx(TEST_LABEL)
    test_data = test.reshape(test.shape[0], -1)
    
    results = []
    epoch = 500
    network.learn_rate = 1e-3
    for i in tqdm(range(epoch)):
        result = network.train_bgd_new(train_data=data)
        result["accuracy"] = evaluate(network.classify(test_data), test_label)
        results.append(result)
        # if i == 10:
        #     print(results)
        #     exit()

    # epoch = 500
    # network.learn_rate = 0.0005
    # for i in tqdm(range(epoch)):
    #     result = network.train_bgd_new(train_data=data, test_data=test_data, test_label=test_label)
    #     results.append(result)
        
    with open("r_28_16_e500.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r)+"\n")
    
    # loss_log = network.train_bgd(data, test_data=test_data, test_label=test_label, epoch=5000)
    # loss_log = loss_log[1:]
    # with open("28_24_16_8_e5000.txt", "w") as f:
    #     f.write(str(loss_log))
        
    # draw_plt("28_24_16_8_e5000.txt")
