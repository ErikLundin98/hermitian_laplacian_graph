# Implementation of "Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian"

This repository contains an unofficial implementation of

> *Graph signal processing for directed graphs based on the Hermitian Laplacian.*
> 
> Furutani, S., Shibahara, T., Akiyama, M., Hato, K., & Aida, M.
> 
> Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
> 
> [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-46150-8_27)


For details, please see the original paper, or [this brief explanation](./docs/README.pdf). A simple use case is provided in [demo.ipynb](./demo.ipynb)

Embeddings are generated for a networkx graph through the ```get_embeddings``` function:

```python
embeddings = get_embeddings(
    G, # networkx graph
    S, # list of scales
    T, # list of "times"
    q, # rotation parameter
    kernel, # a kernel function. Can be a custom function or lib.heat_kernel | low_pass_filter_kernel
    **kernel_args # parameters supplied to the kernel
)
```
An example using the same parameters as the authors, with an undirected and unweighted graph:
```python
from lib import get_embeddings, low_pass_filter_kernel

import numpy as np
import networkx as nx

G = nx.karate_club_graph()
# parameters used in paper
S = np.arange(2.0, 20.1, 0.1) 
T = np.arange(1, 11, 1) 
q = 0.02
kernel = low_pass_filter_kernel
c = 2

embeddings = get_embeddings(
    G=G,
    S=S,
    T=T,
    q=q,
    kernel=kernel,
    c=2
)

```


See [this youtube clip](https://www.youtube.com/watch?v=S4QZiUPJkRI) for a brief explanation of how GraphWave works - they use the graph Laplacian, which only works for undirected graphs and is replaced by the Hermitian Laplacian by Furutani et al.

See [Wikipedia](https://en.wikipedia.org/wiki/Laplacian_matrix#Symmetric_Laplacian_for_a_directed_graph) for an explanation of laplacian matrices