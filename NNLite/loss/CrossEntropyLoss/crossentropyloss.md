# Cross-entropy Loss

Cross entropy loss function is commonly used for categorical classification tasks. It can be defined as the negative multinomial log likelihood

$$E=-\sum_{i=1}^n \sum_{j=1}^m Y_{ij}log\left(Pr(y_{i}=j | x_i))\right)$$

Where Y is the true label matrix, and $Pr(y_{i}=j | x_i)$ is the predicted probability of sample i belonging to class j. As the classes are mutually exclusive, $Y_{i.}$ is a one-hot vector. Consequently, only the terms $log\left(Pr(y_{i}=j | x_i))\right)$ where j corresponds to the true class of sample i contribute to the loss. Let us notate the class probabilities with matrix L: $L_{ij}=log\left(Pr(y_{i}=j | x_i))\right)$.
Now the loss can be expressed in the form

$$E=-\sum_{i, j}\left(Y \odot L \right)_{ij}$$
## Forward pass


