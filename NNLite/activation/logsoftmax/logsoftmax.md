# LogSoftMax

LogSoftMax is an activation function defined by the formula

$$LogSoftMax(x_i)=log(\frac{e^{x_i}}{\sum_{j}^m{e^{x_j}}})$$

It computes the natural logarithm of the softmax function, which maps an input vector x $\in \mathbb{R}^m$ to a normalized probability distribution. Consequently, the output can be interpreted as log-probabilities over m mutually exclusive outcomes, making it suitable for multi-class classification.

## Forward pass
The LogSoftMax is calculated separately for each row of the input matrix X.

$$Z_{ij}=LogSoftMax(X_{ij})=log(\frac{e^{X_{ij}}}{\sum_{k=1}^m{e^{X_{ik}}}})$$
$$log(\frac{e^{X_{ij}}}{\sum_{k=1}^m{e^{X_{ik}}}})=X_{ij} - log(\sum_{k=1}^m(e^{X_{ik}}))$$

## Backward pass

$$\frac{\partial E}{\partial X_{ij}} = \frac{\partial E}{\partial Z}\frac{\partial Z}{\partial X_{ij}}$$
