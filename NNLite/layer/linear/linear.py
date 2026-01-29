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
        self.W = np.random.randn(f, m) * np.sqrt(2 / f)
        self.b = 0.05*np.random.rand(1, m)
        self.gradW = np.zeros((f, m))
        self.gradB = np.zeros((1, m))
        self.X=None

    def forward(self, X):
        """
        calculates the forward pass for this layer,and calculates 
        the gradient of the weight matrix and bias vector wrt. The raw output

        input:
            X: a matrix of inputs, where samples are on rows and features are on columns.
        """

        Z = X @ self.W + self.b
        self.X=X
        return Z

    def backward(self, gradZ):
        """
        Calculate the gradient of the loss wrt. the weight matrix, biases, 
        and the inputs from the last forward pass. gradX will be used as gradZ by
        the previous function in the sequence.
        """
        self.gradW[:]=self.X.T @ gradZ
        self.gradB[:]=np.sum(gradZ, axis=0, keepdims=True)
        gradX=gradZ@self.W.T
        return gradX









    

    


