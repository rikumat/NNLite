# Cross-entropy Loss

Cross entropy loss function is commonly used for categorical classification tasks. It can be defined as the negative multinomial log likelihood

$$E=-\sum_{i=1}^n \sum_{j=1}^m Y_{ij}log\left(Pr(y_{i}=j | x_i))\right)$$

Where Y is the true label matrix, and $Pr(y_{i}=j | x_i)$ is the predicted probability of sample i belonging to class j. As the classes are mutually exclusive, $Y_{i.}$ is a one-hot vector. Consequently, only the terms $log\left(Pr(y_{i}=j | x_i))\right)$ where j corresponds to the true class of sample i contribute to the loss. Let us notate the class probabilities with matrix L: $L_{ij}=log\left(Pr(y_{i}=j | x_i))\right)$.
Now the loss can be expressed in the form

$$E=-\sum_{i, j}\left(Y \odot L \right)_{ij}$$
## Forward pass
The input to the loss function is a matrix of logits Z, where $Z_{ij}$ corresponds to the score of the j:th output of the i:th sample. These outputs are then normalized for each sample, forming n probability distributions. THe normalization is done using the softmax activation function

$$Pr(y_{i}=j | x_i)=\frac{e^{Z_{ij}}}{\sum_{k=1}^N e^{Z_{ik}}}$$

which gives

$$log\left(Pr(y_{i}=j | x_i)\right) = Z_{ij}-log(\sum_{k=1}^N e^{Z_{ik}})$$

For large logits $Z_{ik}$, the exponents can overflow and underflow, resulting in infinities or zeroes. To overcome this, it is common to subtract the maximum $Z_{ik}$ from the exponents in the sum for numerical stability. Let $Z_{max}=max(Z_i.)$


$$log\left(Pr(y_{i}=j | x_i)\right) = Z_{ij}-log(\sum_{k=1}^N e^{Z_{ik}}\frac{e^{Z_{max}}}{e^{Z_{max}}})$$

$$log\left(Pr(y_{i}=j | x_i)\right) = Z_{ij}-log(e^{Z_{max}}\sum_{k=1}^N e^{Z_{ik}-Z_{max}})$$

$$log\left(Pr(y_{i}=j | x_i)\right) = Z_{ij}-Z_{max}-log(\sum_{k=1}^N e^{Z_{ik}-Z_{max}})$$
