# Rectified linear unit (ReLU)

ReLU is a piecewise linear activation function defined as

$$ReLU(x)=x^+ = max(0, x)$$

# Forward pass

ReLU is applied element-wise to the input, $Z_{ij}=ReLU(X_{ij})$.

# Backward pass

$$\frac{\partial E}{\partial X_{ij}}=\frac{\partial E}{ \partial Z_{ij}}\frac{\partial Z_{ij}}{\partial X_{ij}}$$

$$
\frac{\partial Z_{ij}}{\partial X_{ij}} =
\begin{cases} 
1 & \text{if } X_{ij} > 0 \\
0 & \text{if } X_{ij} \le 0
\end{cases}
$$

which gives the gradient

$$
({\nabla}_XE)_{ij} =
\begin{cases} 
({\nabla}_ZE)_{ij} \ & \text{if } X_{ij} > 0 \\
0 & \text{if } X_{ij} \le 0
\end{cases}
$$




