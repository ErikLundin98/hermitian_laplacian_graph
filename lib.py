import networkx as nx
import numpy as np

def get_adj(G:nx.Graph) -> np.matrix:
    return nx.to_numpy_array(G)

def gamma(A:np.matrix, i, j, q) -> np.float32:
    """
    Input:
    A: weighted or unweighted adjacency matrix of a directed graph
    i: row
    j: column
    q: parameter
    Returns:

    """
    return np.exp(1j * np.pi * q * (A[i, j] - A[j, i]))

def degree_matrix(A:np.matrix):
    return np.diag(np.sum(A, axis=1))

def hermitian_laplacian(G:nx.DiGraph, q, degree_normalize=False) -> np.ndarray:
    assert isinstance(G, nx.DiGraph), "method is designed for directed graphs"
    A = get_adj(G)
    A_s = (A + A.T)/2
    D = degree_matrix(A_s)
    Gamma = np.zeros(A.shape)
    for i, j in zip(*A.nonzero()):
        Gamma[i, j] = gamma(A, i, j, q)

    Gamma[Gamma == 0] = 1

    L_q = D - np.multiply(Gamma,A_s)
    
    # if degree_normalize:
    #     inv_D = 
    #     L_q = 
    return L_q
