# Binary cross entropy

Binary cross entropy loss is a loss function used in binary classification problems. It is defined as

$$ BCE = -\frac{1}{NM}\sum_{i=1}^N \sum_{j=1}^M [Y_{ij} log(P_{ij}) + (1-Y_{ij})log(1-P_{ij})]$$

Where $Y_{ij}$ is the true label for the j:th observation of the i:th input (1 for true, 0 for false).

and $P_{ij}$ is the predicted probability $Pr(Y_{ij}=1)$

