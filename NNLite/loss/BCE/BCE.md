# Binary cross-entropy 

Binary cross-entropy loss is a loss function used in binary classification problems. It is applicable when the output of the network for a single input is one or more probabilities for a specific label (usually 1) for a binary outcome. More precisely, the output is a vector z, where $z_i = Pr(y_i=1)$, and $y_i$ is the true label for that output. Binary cross-entropy loss is the average negative log-likelihood for all $y_i$ 

$$Pr(y_i=1|\theta) = z_i \quad and \quad Pr(y_i=0|\theta) = 1-z_i$$ 

Which gives the likelihood function 

$$\mathcal{L}(\theta | y_i) = z_i^{y_i}*(1-z_i)^{1-y_i}$$

and the log likelihood

$$log\mathcal{L}(\theta | y_i) = y_ilog(z_i) + (1-y_i)log(1-z_i)$$

This gives the loss

$$ E=-\frac{1}{M}\sum_{i=1}^M[y_ilog(z_i) + (1-y_i)log(1-z_i)]$$

