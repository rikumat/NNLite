import numpy as np

class Linear:
    """
    Linear layer of a feedforward network
    """
    def __init__(self, f, m):
        """
        f: the number of expected input features
        m: the number of neurons on the layer
        """
        self.W = 0.01*np.random.rand(f, m)
        self.b = 0.01*np.random.rand(1, m)
        self.gradW = np.zeros((f, m))
        self.gradB = np.zeros((1, m))

    def forward(self, X):
        """
        calculates the forward pass for this layer,and calculates 
        the gradient of the weight matrix and bias vector wrt. The raw output

        input:
            X: a matrix of inputs, where samples are on rows and features are on columns.
        """

        out = X @ self.W + self.b
        return out





    

    


