# Binary cross-entropy 

Binary cross-entropy (BCE) loss is a loss function used for binary classification problems. It is applicable when the network outputs, for a single input, one or more probabilities representing independent binary labels. More precisely, the output is a vector z, where $z_i = Pr(y_i=1)$, and $y_i$ is the true label for that output. Binary cross-entropy loss is the average negative log-likelihood for all $y_i$. 

$$Pr(y_i=1|\theta) = z_i \quad and \quad Pr(y_i=0|\theta) = 1-z_i$$ 

Which gives the likelihood function 

$$\mathcal{L}(\theta | y_i) = z_i^{y_i}(1-z_i)^{1-y_i}$$

and the log likelihood

$$log\mathcal{L}(\theta | y_i) = y_ilog(z_i) + (1-y_i)log(1-z_i)$$

This gives the loss

$$ E=-\frac{1}{M}\sum_{i=1}^M[y_ilog(z_i) + (1-y_i)log(1-z_i)]$$

## Forward pass

When using batched training, the losses for individual samples are usually combined by averaging them. This results in the final loss function

$$ E=-\frac{1}{MN}\sum_{i=1}^N\sum_{j=1}^M[Y_{ij}log(Z_{ij}) + (1-Y_{ij})log(1-Z_{ij})]$$

Where
* $Y_{ij}$ is the true label for output j in sample i
* $Z_{ij}$ is the predicted probability of the true label $Y_{ij}$ being 1
* N is the number of samples
* M is the number of outputs in each sample

## Backward pass
Since the loss is a sum of terms each depending only on one Z_{ij}, the calculation of partial derivatives is straightforward. Specifically, we get

$$ \frac{E}{Z_{ij}}=-\frac{1}{MN}[Y_{ij}log(Z_{ij}) + (1-Y_{ij})log(1-Z_{ij})]$$

resulting in 

$$ \frac{E}{Z_{ij}}=-\frac{1}{MN}\left(\frac{Y_{ij}}{Z_{ij}} - \frac{1-Y_{ij}}{1-Z_{ij}}\right)$$

