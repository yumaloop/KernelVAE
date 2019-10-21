import numpy as np

class NNArccosineKernel():
    """
    depth: int
        network depth (total num of layers)
    input_dims: float
        num of units in each layer (len(input_dims) == depth)
    wvar: float 
        variance of weights
    """
    def __init__(self, depth=3, input_dims=[784, 100, 100], wvar=1.0):
        self.wvar = wvar
        self.depth = depth
        self.input_dims= input_dims

    def __call__(self, x, y):
        return self._kernel_func(x, y, self.depth)

    def _kernel_func(self, x, y, l):
        if l == 0:
            x_l2norm = np.linalg.norm(x, ord=2)
            y_l2norm = np.linalg.norm(y, ord=2)
            theta = np.arccos(np.dot(x, y) / (x_l2norm * y_l2norm))
            return (x_l2norm * y_l2norm * self._jacobian_func(theta)) / (np.pi * (self.wvar ** self.input_dims[l]))
        else:
            kernel_xy = self._kernel_func(x, y, l-1)
            kernel_xx = self._kernel_func(x, x, l-1)
            kernel_yy = self._kernel_func(y, y, l-1)
            theta = np.arccos(kernel_xy / np.sqrt(kernel_xx * kernel_yy))
            return np.sqrt(kernel_xx * kernel_yy) * self._jacobian_func(theta) / (np.pi * (self.wvar ** self.input_dims[l-1]))

    def _jacobian_func(self, t):
        return np.sin(t) + (np.pi - t) * np.cos(t)


class RBFKernel():
    """
    """
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x, y):
        return np.exp( (-1. / (self.sigma ** 2) ) * np.linalg.norm(x-y, ord=2) ** 2)

    def derivatives(self, x1, x2):
        dif_sigma = np.exp( (-1. / (self.sigma ** 2) ) *  (x - y) ** 2) * ( np.linalg.norm(x-y, ord=2) ** 2 ) / ( self.sigma ** 3)
        return dif_sigma

    def update_sigma(self, update):
        self.sigma += update
