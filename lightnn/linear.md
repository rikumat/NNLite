# Linear layer construction

## Initialization
When creating a layer, the class needs to know the number of neurons on the layer, and the number of features in the inputs. We denote these with the constants m and f. These constants are used to create the weight matrix and bias vector. Now

$W\in \mathbb{R}^{fxm}$, $X\in \mathbb{R}^{nxf}$, $b \in \mathbb{R}^m$

Where f is the number of input features, m is the number of neurons, and n is the number of inputs.

$$(XW)_{ij}=X_{i.}W_{.j}$$

We expect as an input a matrix X where samples are on rows and features are on columns. As such, $X_{ij}$ denotes the j:th feature of the i:th input. To transform the input linearly, The output of the layer is calculated with the expression $Z=XW+b$, such that $Z_{ij}$ is the output of the j:th neuron for the i:th input. The bias vector is added to every row of XW.

