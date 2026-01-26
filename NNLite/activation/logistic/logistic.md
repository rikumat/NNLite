# Logistic activation

Logistic activation is a sigmoid function defined as

$$ {\sigma}(x) = \frac{1}{1+e^{-x}} $$

It maps outputs to a value between 0 and 1, which makes it suitable for representing probabilities. 


## forward pass
Sigmoid is applied to the input element-wise. The result Z is saved in the state of the class for later use during backpropagation.

$$ Z_{ij} = {\sigma}(X_{ij}) $$

## Backward pass

$$ \frac{\partial E}{\partial X_{ij}} = \frac{\partial E}{\partial Z_{ij}}\frac{\partial Z_{ij}}{\partial X_{ij}} $$

$$\frac{\partial Z_{ij}}{\partial X_{ij}} = {\sigma}(X_{ij})(1-{\sigma}(X_{ij}))$$

$$ {\nabla}_XE = \nabla_ZE \odot Z \odot (1-Z) $$
