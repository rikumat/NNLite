# Linear layer construction

## Overview
A linear (or fully connected) layer is a specific type of layer used in neural networks. In these layers, all inputs are connected to all outputs; hence the name “fully connected.” Each neuron in a linear layer calculates its output based on the weighted sum of its inputs and a bias term. Consequently, the number of outputs from a linear layer is equal to the number of neurons in that layer. These outputs can then be used as inputs to the next layer.
Full connectivity offers favorable attributes, such as the ability to compute the layer’s output using matrix–vector multiplication. In the case of a single input vector x with f features, the weighted sum can be computed by multiplying x by a weight matrix W, where each column of W corresponds to the weights associated with a specific neuron. The output can then be completed by adding the bias terms, which can be done by summing a vector of biases to the result of the multiplication.

## Initialization
When creating a layer, the class needs to know the number of neurons on the layer, and the number of features in the inputs. We denote these with the constants m and f. These constants are used to create the weight matrix and bias vector. Now

$$W\in \mathbb{R}^{fxm}, \ X\in \mathbb{R}^{nxf},\ b \in \mathbb{R}^m$$

Where f is the number of input features, m is the number of neurons, and n is the number of inputs.

$$(XW)_{ij}=X_{i.}W_{.j}$$

We expect as an input a matrix X where samples are on rows and features are on columns. As such, $X_{ij}$ denotes the j:th feature of the i:th input. To transform the input linearly, The output of the layer is calculated with the expression $Z=XW+b$, such that $Z_{ij}$ is the output of the j:th neuron for the i:th input. The bias vector is added to every row of XW.

## Forward pass
To accommodate later gradient calculation during backpropagation, each layer needs to keep track of its local gradient.

$$\frac{\partial E}{\partial W_{ij}} = \sum_{l,k}(\frac{\partial E}{\partial Z_{lk}}\frac{\partial Z_{lk}}{\partial W_{ij}})$$

The elements in the sum are nonzero only when 

$$\frac{\partial Z_{lk}}{\partial W_{ij}}\ne 0 \ or \ equivalently \ j=k$$

Which reduces the sum to 

$$\frac{\partial E}{\partial W_{ij}} = \sum_{l}(\frac{\partial E}{\partial Z_{lj}}\frac{\partial Z_{lj}}{\partial W_{ij}})$$

as 

$$Z_{lj}=\sum_{i}(X_{li}W_{ij})+b_j$$

We have 

$$ \frac{\partial Z_{lj}}{\partial W_{ij}}=X_{li} $$

$$ \frac{\partial E}{\partial W_{ij}} = \sum_{l}(\frac{\partial E}{\partial Z_{lj}}X_{li}) = \sum_{l}(X^{T}_{il}\frac{\partial E}{\partial Z_{lj}})$$

Now

$$ \frac{\partial E}{\partial W_{ij}} = (X^T{\nabla}_{Z}E)_{ij} \ and \ {\nabla}_{W}E = X^T{\nabla}_{Z}E $$

To be able to calculate this gradient during backward pass, we store the input X as a class attribute. The gradient ${\nabla}_{Z}E$ will be provided separately by the subsequent function, whether it be the next layer, activation function, or the loss function.

## Backward pass
After forwarding the output and saving the inputs of the current batch, the next step is to calculate the gradient of the loss with respect to the layer's weights. The backward method takes as an argument the gradient ${\nabla}_{Z}E$ and calculates the gradient of the weights with the formula derived above. The gradient is then saved as a class attribute for further use by an optimizer.

as

$$Z_{lj}=\sum_{i}(X_{li}W_{ij})+b_j$$

$$\frac{\partial E}{\partial b_j} = \sum_{l}(\frac{\partial E}{\partial Z_{lj}}\frac{\partial Z_{lj}}{\partial b_j})= \sum_{l}(\frac{\partial E}{\partial Z_{lj}})$$

The gradient of F wrt. the bias terms is the column-wise sum of ${\nabla}_ZE$.

$${\nabla}_{b}E=\sum_{l}{({\nabla}_ZE)_{l.}}$$

The final step is to calculate the gradient of E wrt. X.

$$\frac{\partial E}{\partial X_{ij}} = \sum_{l}({\nabla}_ZE_{il} \frac{\partial Z_{il}}{\partial X_{ij}})=\sum_{l}({\nabla}_ZE_{il}W_{jl})=({\nabla}_ZE W^T)_{ij} $$

which results in 

$${\nabla}_XE={\nabla}_ZE W^T$$

The last gradient is returned to be used by the previous layer during backpropagation.









