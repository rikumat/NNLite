import numpy as np

class L2:
    def __init__(self, c=0.01):
        self.c = c

    def forward(self, params):
        return (1/2)*self.c * sum(np.sum(p**2) for p in params)

    def backward(self, params):
        return params*self.c
