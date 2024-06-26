import numpy as np
# out_grad is going to be numpy array. This is because we don't want gradient to itself have
# a gradient. We are not worried about double differentiation.

class Tensor:
    def __init__(self, array, children=(), op="") -> None:
        """
        tensors are numpy arrays
        """
        self.array = array
        # tuple of tensors
        self.children = children
        self.grad = 0 # we are adding, so can't start at 1
        # how he does it, rather than creating an operation class
        # add underscore because we want atribute to be different than the actual backward function
        self._backward = lambda: 4
        self.op = op

    def __add__(self, other):
        # self.__add__(other)
        # (self, other) = (Tensor, Tensor)
        if isinstance(other, Tensor):
            out = Tensor(self.array + other.array, (self, other), "+")
        else:
            out = Tensor(self.array + other, (self, ), "+")
        # create backward function
        # self  ---> 
        #            +   --> out
        # other --->
        def _backward():
            # imagine this was the last operation, the out.grad would be known (just 1)
            # and so the grads to computer are self.grad and other.grad. Even if this
            # wasn't  the last operation and was in the middle, we would still know what
            # out.grad is because of idea above being trickled down through the network
            self.grad += out.grad
            if isinstance(other, Tensor):
                other.grad += out.grad
        # want to make out.backward be the backward() function, which will then update necessary
        # gradiants 
        out._backward = _backward # can't do backward() because function returns none
        # out will have property backward which we have set
        # want to return out after defining function
        return out
    
    def __radd__(self, other): # other + self
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.array * other.array, (self, other), "*")
        
        def _backward():
            self.grad += out.grad * other.array
            other.grad += out.grad * self.array
        out._backward = _backward
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __pow__(self, power):
        if isinstance(power, Tensor):
            out = Tensor(self.array ** power.array, (self, power), "**")
        else:
            out = Tensor(self.array ** power, (self, ), "**")
        
        def _backward():
            a = self.array
            if isinstance(power, Tensor):
                b = power.array
                power.grad += out.grad * ((a ** b) * np.log(a))
            else:
                b = power
            self.grad += out.grad * (b * (a ** (b - 1)))
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def log(self):
        out = np.log(self.array)
        out = Tensor(out, (self, ), "log")

        def _backward():
            self.grad += out.grad / self.array
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = np.exp(self.array)
        out = Tensor(out, (self,), "exp")

        def _backward():
            grad  = out.grad * np.exp(self.array)
            self.grad += grad
        out._backward = _backward
        return out


    def reshape(self, shape):
        out = np.reshape(self.array, shape)
        out = Tensor(out, (self,), "reshape")

        def _backward():
            if isinstance(out.grad, float):
                out.grad = np.full(out.array.shape, out.grad)
            self.grad += np.reshape(out.grad, self.array.shape)
        
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-1 * self.array, (self, ), "neg") 
        def _backward():
            self.grad += (-1 * out.grad)
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __matmul__(self, other):
        other = Tensor(other) if isinstance(other, np.ndarray) else other
        out = Tensor(self.array @ other.array, (self, other), "@")
        
        def _backward():
            b = other.array
            a = self.array
            # calculating grad of transpose within grad function does not make sense
            if isinstance(out.grad, float):
                out.grad = np.full(out.array.shape, out.grad)
            
            grad_self = out.grad @ b.T
            if grad_self.shape != a.shape:
                grad_self =  np.sum(grad_self, axis = tuple(range(len(grad_self.shape) - len(a.shape))))
            self.grad += grad_self
            
            grad_other = a.T @ out.grad
            if grad_other.shape != b.shape:
                grad_other = np.sum(grad_other, tuple(range(len(grad_other.shape) - len(b.shape))))
            other.grad += grad_other

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self * other
    
    def relu(self):
        out = np.maximum(self.array, 0)
        out = Tensor(out, (self, ), "relu")
        def _backward():
            self.grad += ((self.array > 0) * out.grad)
        out._backward = _backward
        return out

    # http://coldattic.info/post/116/
    def BroadcastTo(self, shape):
        # self.shape is m x 1
        if self.array.shape[0] == shape[0]:
            ones_array = np.ones((1, shape[1]))
            return self.__matmul__(Tensor(ones_array))
        self.array = self.array.reshape(-1, 1)
        ones_array = np.ones((shape[0], 1))
        # explictly call transpose
        self = self.transpose(axes=None)
        return Tensor(ones_array).__matmul__(self)


    def summation(self, axes):
        out = np.sum(self.array, axes)
        out = Tensor(out, (self,), "sum")

        def _backward():
            if isinstance(out.grad, float):
                out.grad = np.broadcast_to(out.grad, self.array.shape)
            if out.grad.ndim == 0:
                out.grad = out.grad.reshape((-1,1))
            if axes is not None and 0 in axes:
                out.grad = np.transpose(out.grad)
            ones = np.ones_like(self.array)
            if out.grad.shape != ones.shape:
                self.grad += np.broadcast_to(out.grad.reshape(-1, 1), self.array.shape)  * np.ones_like(self.array)
            else:
                self.grad += out.grad * np.ones_like(self.array)

        out._backward = _backward
        return out

    def transpose(self, axes):
        if axes:
            out = np.swapaxes(self.array, *axes)
        else:
            out = np.swapaxes(self.array, self.array.ndim - 2, self.array.ndim - 1)
        out = Tensor(out, (self,), "transpose")

        def _backward():
            self.grad += out.grad.transpose(axes)
        
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(array={self.array}, grad={self.grad})"
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
