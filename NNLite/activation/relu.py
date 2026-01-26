import numpy as np

class Relu:
    def __init__(self):
        self.X=None
        pass

    def forward(self, X):
        self.X=X
        return np.maximum(X, 0)
    
    def backward(self, gradZ):
        gradX=gradZ*(self.X>0)
        return gradX

