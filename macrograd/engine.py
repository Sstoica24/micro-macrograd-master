
import numpy as np
# would be a tensor that has same thing but vectorized.
class Tensor:
    def __init__(self, array, children=(), op="", batch_size=1) -> None:
        """
        tensors are numpy arrays
        """
        self.array = array
        self.batch_size = batch_size
        # tuple of tensors
        self.children = children
        self.grad = 0 # we are adding, so can't start at 1
        # how he does it, rather than creating an operation class
        # add underscore because we want atribute to be different than the actual backward function
        self._backward = lambda: None
        self.op = op

    def __add__(self, other):
        # self.__add__(other)
        # (self, other) = (Tensor, Tensor)
        if isinstance(other, Tensor):
            print("I am here")
            out = Tensor(self.array + other.array, (self, other), "+", batch_size=self.batch_size)
        else:
            out = Tensor(self.array + other, (self, ), "+", batch_size=self.batch_size)
        # create backward function
        # self  ---> 
        #            +   --> out
        # other --->
        def _backward():
            # imagine this was the last operation, the out.grad would be known (just 1)
            # and so the grads to computer are self.grad and other.grad. Even if this
            # wasn't  the last operation and was in the middle, we would still know what
            # out.grad is because of idea above being trickled down through the network
            self.grad += 1/self.batch_size * out.grad
            if isinstance(other, Tensor):
                print("bacth_size", self.batch_size)
                print("out.grad", out.grad)
                other.grad += 1/self.batch_size * out.grad
        # want to make out.backward be the backward() function, which will then update necessary
        # gradiants 
        out._backward = _backward # can't do backward() because function returns none
        # out will have property backward which we have set
        # want to return out after defining function
        return out
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.array * other.array, (self, other), "*", batch_size=self.batch_size)
        else:
            out = Tensor(self.array * other, (self, ),  "*", batch_size=self.batch_size)
        
        def _backward():
            self.grad += 1/self.batch_size * out.grad * other.array
            if isinstance(other, Tensor):
                other.grad += 1/self.batch_size * out.grad * self.array
        out._backward = _backward
        return out

    def __pow__(self, power):
        if isinstance(power, Tensor):
            out = Tensor(self.array ** power.array, (self, power), "**", batch_size=self.batch_size)
        else:
            out = Tensor(self.array ** power, (self, ), "**", batch_size=self.batch_size)
        
        def _backward():
            self.grad += 1/self.batch_size * out.grad * (power.array * self ** (power.array - 1))
            if isinstance(power, Tensor):
                power.grad += 1/self.batch_size * out.grad * (self**power) * np.log(self.array)
        out._backward = _backward
        return out
    
    def __relu__(self):
        out = Tensor(np.max(0, self.array), (self, ), "relu", batch_size=self.batch_size)
        def _backward():
            self.grad += 1/self.batch_size * ((self.array > 0) * out.grad)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-1 * self, (self, ), "neg", batch_size=self.batch_size) 
        def _backward():
            self.grad += 1/self.batch_size * (-1 * out.grad)
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __matmul__(self, other):
        out = Tensor(self.array @ other.array, (self, other), "@", batch_size=self.batch_size)
        def _backward():
            b = other.array
            a = self.array
            grad_self = out.grad @ b.Transpose(axes=None)
            grad_other = 1/self.batch_size * a.Transpose(axes=None) @ out.grad

            if grad_self.shape != a.shape:
                grad_self =  grad_self.Summation(tuple(range(len(grad_self.shape) - len(a.shape))))
            if grad_other.shape != b.shape:
                grad_other = grad_other.Summation(tuple(range(len(grad_other.shape) - len(b.shape))))
            
            self.grad += 1/self.batch_size * grad_self
            other.grad += 1/self.batch_size * grad_other

        out._backward = _backward
        return out

    def BroadcastTo(self, shape):
        out = np.broadcast_to(self.array, shape)
        out = Tensor(out, (self,), "broadcast", batch_size=self.batch_size)

        def _backward():
            a = self.array
            array_out = out.grad
            num_broadcasted_dims = len(array_out.shape) - len(a.shape)
            # Sum along the added dimensions to get the gradient of the smaller array
            #grad_a = summation(out_grad, tuple(range(num_broadcasted_dims)))
            grad_a = np.ones_like(a.shape) * array_out
            self.grad += 1/self.batch_size * grad_a
        
        out._backward = _backward
        return out

    def Summation(self, axes):
        out = np.sum(self.array, axis=axes)
        out = Tensor(out, (self,), "summation", batch_size=self.batch_size)

        def _backward():
            array_out = out.grad
            if array_out.ndim == 0:
                array_out.reshape((-1,1))
            
            if 0 in self.axes:
                array_out.Transpose(axes=None)
            
            self.grad += 1 / self.batch_size * array_out * np.ones_like(self)
        out._backward = _backward
        return out
    
    def Transpose(self, axes):
        if axes:
            out = np.swapaxes(self.array, *axes)
        else:
            a = self.array
            out = np.swapaxes(a, a.ndim - 2, a.ndim - 1)
        out = Tensor(self, (self,), ".T", batch_size=self.batch_size)

        def _backward():
            array_out = out.grad
            self.grad += 1/ self.batch_size * (array_out.Transpose(axes))
        out._backward = _backward
        return out


    def __repr__(self):
        return f"Tensor(array={self.array}, grad={self.grad})"
    
    # assumption is that this is called on the out tensor, which is why
    # it is ok to put it in the class
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
        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
            