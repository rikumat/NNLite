class Sequential:
    def __init__(self, *functions):
        self.functions = functions

    def forward(self, X):
        for func in self.functions:
            X=func.forward(X)
        return X
    
    def backward(self, gradZ):
        for func in self.functions[::-1]:
            gradZ = func.backward(gradZ)
        
        




