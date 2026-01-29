# Binary cross-entropy 

Binary cross-entropy (BCE) loss is a loss function used for binary classification problems. It is applicable when the network outputs, for a single input x, one or more values representing independent probabilities of binary labels. More precisely, the output is a vector z, where $z_i = Pr(y_i=1|\theta, x_i)$, and $y_i$ is the true label for that output. Binary cross-entropy loss is defined by the formula

$$E=-\frac{1}{M}\sum_{i=1}^M[y_ilog(z_i) + (1-y_i)log(1-z_i)]$$

This can be seen as the negative mean log likelihood of observing y, given parameters $\theta$ and input x.

$$Pr(y_i=1|\theta, x_i) = z_i \quad and \quad Pr(y_i=0|\theta, x_i) = 1-z_i$$ 

Which gives the likelihood function 

$$\mathcal{L}(\theta | y, x) = \prod_{i=1}^M z_i^{y_i}(1-z_i)^{1-y_i}$$

and the log likelihood

$$\log\mathcal{L}(\theta | y, x) =\sum_{i=1}^M [y_ilog(z_i) + (1-y_i)log(1-z_i)]$$

Which further leads to the loss defined above.

## Forward pass

When using batched training, the losses for individual samples are usually combined by averaging them. This results in the final loss function

$$ E=-\frac{1}{MN}\sum_{i=1}^N\sum_{j=1}^M[Y_{ij}log(Z_{ij}) + (1-Y_{ij})log(1-Z_{ij})]$$

Where
* $Y_{ij}$ is the true label for output j in sample i
* $Z_{ij}$ is the predicted probability of the true label $Y_{ij}$ being 1
* N is the number of samples
* M is the number of outputs in each sample

## Backward pass
Since the loss is a sum of terms each depending only on one $$Z_{ij}$$, the calculation of partial derivatives is straightforward. Specifically, we get

$$ \frac{\partial E}{\partial Z_{ij}}=-\frac{1}{MN}\frac{\partial}{\partial Z_{ij}}[Y_{ij}log(Z_{ij}) + (1-Y_{ij})log(1-Z_{ij})]$$

resulting in 

$$ \frac{\partial E}{ \partial Z_{ij}}=-\frac{1}{MN}\left(\frac{Y_{ij}}{Z_{ij}} - \frac{1-Y_{ij}}{1-Z_{ij}}\right)$$

giving the gradient

$$ \nabla_ZE=-\frac{1}{MN}\left(\frac{Y}{Z} - \frac{1-Y}{1-Z}\right)$$





