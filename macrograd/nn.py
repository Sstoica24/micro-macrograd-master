import random
import numpy as np
from macrograd.engine import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    # adding @property above paramters def would allow us to call paramters by doing just model.parameters
    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, num_inputs, num_outputs, nonlin=True):
        # self.biases = Tensor(np.zeros((num_outputs, 1)))
        self.biases = Tensor(np.zeros((1, num_outputs)))
        # if I don't multiply by stddev, weights get way too large, causing softmax
        # to become way to large
        stddev = np.sqrt(2 / num_inputs)  # Or adjust scaling factor as needed
        self.weights = Tensor(np.random.randn(num_inputs, num_outputs) * stddev)
        # print(f"{self.weights= }")
        # print(f"{self.biases= }")
        self.nonlin = nonlin
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
    
    def __call__(self, X):
        mult = X @ self.weights
        y = mult + self.biases.BroadcastTo((mult.array.shape))
        return y.relu() if self.nonlin else y

    def parameters(self):
        self.biases.array = self.biases.array.reshape(-1, 1)
        # returning tensors
        return [self.weights, self.biases]
    
    def __repr__(self) -> str:
        return f"weights of the layer are {self.weights}, and the bias is {self.biases}"

class MLP_macro(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
