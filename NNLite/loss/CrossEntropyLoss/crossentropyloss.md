# Cross-entropy Loss

Cross entropy loss function is commonly used for categorical classification tasks. It can be defined as the negative multinomial log likelihood

$$E=-\sum_{i=1}^n \sum_{j=1}^m Y_{ij}log\left(Pr(y_{i}=j | x_i))\right)$$

Where Y is the true label matrix, and $Pr(y_{i}=j | x_i)$ is the predicted probability of sample i belonging to class j. As the classes are mutually exclusive, $Y_{i.}$ is a one-hot vector. Consequently, only the terms $log\left(Pr(y_{i}=j | x_i))\right)$ where j corresponds to the true class of sample i contribute to the loss. Let us notate the class probabilities with matrix $L_{ij}=log\left(Pr(y_{i}=j | x_i))\right)$.
Now the loss can be expressed in the form

$$E=-\sum_{i, j}\left(Y \odot L \right)_{ij}$$
## Forward pass
The input to the loss function is a matrix of logits Z, where $Z_{ij}$ corresponds to the score of the j:th output of the i:th sample. These outputs are then normalized for each sample, forming n probability distributions. THe normalization is done using the softmax activation function

$$Pr(y_{i}=j | x_i)=\frac{e^{Z_{ij}}}{\sum_{k=1}^M e^{Z_{ik}}}$$

For large logits $Z_{ik}$, the exponents can overflow and underflow, resulting in infinities or zeroes. To overcome this, it is common to subtract the maximum of $Z_{i.}$ from the exponents in the sum for numerical stability.


$$L_{ij} = log\left(\frac{e^{Z_{ij}-max(Z_{i.})}}{\sum_{k=1}^M e^{Z_{ik}-max(Z_{i.})}}\right)$$

## Backward pass

$$L_{ij} = Z_{ij}-max(Z_{i.})-log(\sum_{k=1}^M e^{Z_{ik}-max(Z_{i.})})$$

let c be the index of the correct class.

$$\frac{\partial E}{\partial Z_{ic}}=-\frac{\partial L_{ic}}{\partial Z_{ic}}$$
$$\frac{\partial E}{\partial Z_{ic}}= -1+ \frac{e^{Z_{ic}-max(Z_{i.})}}{\sum_{k=1}^M e^{Z_{ik}-max(Z_{i.})}}$$

And for j $\ne$ c we get 

$$\frac{\partial E}{\partial Z_{ij}}=-\frac{\partial L_{ic}}{\partial Z_{ij}}$$

$$\frac{\partial E}{\partial Z_{ij}}=\frac{e^{Z_{ij}-max(Z_{i.})}}{\sum_{k=1}^M e^{Z_{ik}-max(Z_{i.})}}$$

Let us notate the numerically safe softmax values with matrix P. 

$$P_{ij}=\frac{e^{Z_{ij}-max(Z_{i.})}}{\sum_{k=1}^M e^{Z_{ik}-max(Z_{i.})}}$$

As $Y_{i.}$ is a one-hot vector with the nonzero entry corresponding to the correct class, we can express the gradient in the form

$$\nabla_{Z}E=P-Y$$






