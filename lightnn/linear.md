# Linear layer construction

## Initialization

$W\in R^{fxm}$, $X\in R^{nxf}$, $b \in R^{1xm}$

$$(XW)_{ij}=X_{i.}W_{.j}$$

We expect as an input a matrix X where samples are on rows and features are on columns. As such, $X_{ij}$ denotes the j:th feature of the i:th input. To transform the input linearly, The output of the layer is calculated with the expression $Z=XW+b$, such that $Z_{ij}$ is the output of the j:th neuron for the i:th input.


