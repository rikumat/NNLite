class Sequential:
    def __init__(self, *functions):
        self.functions = functions

    def forward(self, X):
        for func in self.functions:
            X=func.forward(X)
        self.Z = X
        return X
    
    def backward(self, loss):
        gradE = loss.backward()
        for func in self.functions[::-1]:
            gradE = func.backward(gradE)

    def params(self):
        parameters = []
        for func in self.functions:
            if hasattr(func, "W"):
                parameters.append({"param": func.W, "grad": func.gradW})
            if hasattr(func, "b"):
                parameters.append({"param": func.b, "grad": func.gradB})
        return parameters


            
        
        




