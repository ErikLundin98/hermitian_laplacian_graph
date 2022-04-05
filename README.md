# Implementation of "Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian

The implementation uses GraphWave, but replaces the wavelet used in GraphWave with 

$$
\boldsymbol{\psi}_{s, i} = \boldsymbol{U\hat{G}_{s}U^* \delta_i}
$$

Here, $\boldsymbol U$ is a matrix where the columns are the eigenvectors of the graph Laplacian $\boldsymbol L_q$

$$
\boldsymbol L_q = \boldsymbol{D} - \boldsymbol{\Gamma}_q \circledcirc \boldsymbol{A}^{(s)}
$$

Where $\boldsymbol{D}$ is the degree matrix of the symmetrized graph $\boldsymbol{G}^{(s)}$, $\boldsymbol{\Gamma}_q$ is the function

$$
\gamma_q(i,j) = \exp(i2\pi q(w_{ij} - w_{ji}))
$$

applied to all elements of the adjacency matrix of $\boldsymbol{G}$. $\circledcirc$ is elementwise multiplication. $*$ denotes the conjugate transpose.

$\hat{G}_{s} = diag(\hat{g}(s\lambda_0), \dotsc , \hat{g}(s\lambda_{N-1}))$ where $\hat{g}(s\cdot)$ is a unique filter kernel, in our implementation either a low-pass filter kernel

$$
\hat{h}(\lambda) = \frac{1}{1+c\lambda}
$$
or the heat kernel 

$$\hat{h}(\lambda) =e^{-s\lambda}$$

Lastly, $\boldsymbol{\delta_i}$ is a vector whose i-th entry is 1 and the others are 0.

After this, we have the wavelets $\boldsymbol \psi$. After this, the authors use the same embedding technique that is used by graphwave:

Given a vector $T$ of d values and a vector $S$ of m values, the embeddings are given by

$$
x_i = [Re(\phi_i(s, t)), Im(\phi_i(s, t))]_{t\in T, s \in S}
$$

Where 
$$
\phi_i(s,t)=\frac{1}{N}\sum_{j=1}^{N}e^{it\psi_{ij}(s)}
$$

This means that phi needs to be calculated for a bunch of s values.