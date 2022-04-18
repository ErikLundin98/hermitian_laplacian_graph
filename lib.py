import networkx as nx
import numpy as np
from numba import jit
from typing import Union
from tqdm.auto import tqdm

"""
This is a non-official implementation of 
"Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian",
SOURCE:
******************************************************************************
FURUTANI, Satoshi, et al. 
Graph signal processing for directed graphs based on the Hermitian Laplacian. 
In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. 
Springer, Cham, 2019. p. 447-463.
******************************************************************************
"""

# Parameters used by Furutani et al.
HERMLAP_S = np.arange(2.0, 20.1, 0.1) 
HERMLAP_T = np.arange(1, 11, 1) 

def get_adj(G:nx.Graph) -> np.matrix:
    """Utility function to extract the adjacency matrix of a networkx graph"""
    return nx.to_numpy_array(G)

@jit
def gamma(A:np.matrix, i, j, q) -> np.float32:
    """
    Input:
    A: weighted or unweighted adjacency matrix of a directed graph
    i: row
    j: column
    q: parameter
    Returns:
    the value of the gamma function
    """
    return np.exp(1j * np.pi * q * (A[i, j] - A[j, i]))

@jit
def degree_matrix(A:np.matrix) -> np.matrix:
    """Returns a degree matrix of a weighted or unweighted adjacency matrix"""
    return np.diag(np.sum(A, axis=1))

@jit
def hermitian_laplacian(A:np.matrix, q) -> np.matrix:
    """Returns the Hermitian Laplacian of a graph as defined in the paper"""

    A_s = (A + A.T)/2
    D = degree_matrix(A_s)
    Gamma = np.zeros(A.shape, dtype=np.complex64)
    for i, j in zip(*A.nonzero()):
        Gamma[i, j] = gamma(A, i, j, q)

    L_q = D - np.multiply(Gamma,A_s)
    
    return L_q

@jit
def low_pass_filter_kernel(x:np.ndarray, c:float=1.0) -> np.ndarray:
    """Low-pass filter kernel"""
    return 1/(1+c*x)

@jit
def heat_kernel(x:np.ndarray) -> np.ndarray:
    """Heat kernel"""
    return np.exp(-x)

@jit
def phi(psi:np.matrix, t:float) -> np.float32:
    """
    Computes an embedding from the wavelet for a node I
    params:
    psi: Wavelets for the graph
    i: index of node
    t: embedding parameter
    """
    return 1/psi.shape[0] * np.sum(
        np.exp(1j * t * psi),
        axis=1
    )

@jit
def determine_proper_S(eigenvalues:np.ndarray) -> np.ndarray:
    """Determine scale parameter values according to graphwave"""
    nu = 0.85
    gamma = 0.95
    denom = np.sqrt(eigenvalues[1]*eigenvalues[-1])
    s_max = -np.log(nu)/denom
    s_min = -np.log(gamma)/denom
    S = np.linspace(s_min, s_max, 2)
    return S

def get_embeddings(A:Union[nx.Graph, np.matrix], S:list[float]=None, T:list[float]=np.arange(0, 101, 2), q=0.5, kernel:callable=low_pass_filter_kernel, progress:bool=False, **kernel_args) -> np.matrix:
    """
    Main function that computes graphwave embeddings from the Hermitian Laplacian
    Can be used with a (un)weighted (di)graph
    params:
    G: Adjacency matrix or networkx graph to extract embeddings from
    S: List of scale parameters. Determines the radius of network neighborhood
    T: List of sampling points
    q: Rotation parameter for the Hermitian Laplacian
    kernel: A kernel callable that can take a one-dimensional numpy array as input and returns a transformed version of the input
    **kernel_args: keyword arguments for the kernel function
    returns:
    a N x 2*|S|*|T| matrix
    if S is not specified, an appropriate range of values will be determined automatically according to the GraphWave paper
    """
    if isinstance(A, nx.Graph):
        A = get_adj(A)
    
    N = A.shape[0]
    if progress: print("computing the hermitian laplacian")
    L_q = hermitian_laplacian(A, q)
    if progress: print("computing eigenvectors and eigenvalues")
    eigenvalues, U = np.linalg.eig(L_q)
    eigenvalues = np.real(np.array(eigenvalues))

    if S is None:
        if progress: print("auto-inferring S")
        S = determine_proper_S(eigenvalues)

    U = np.matrix(U)
    delta = np.eye(N)

    re_embeddings = np.zeros((N, len(S)*len(T)))
    im_embeddings = np.zeros(re_embeddings.shape)

    len_T = len(T)
    print("extracting features")
    for s_idx, s in enumerate(tqdm(S, disable=not progress)):
        G_hat_s = np.diag(kernel(eigenvalues*s, **kernel_args))
        psi = U @ G_hat_s @ U.H @ delta
        for t_idx, t in enumerate(tqdm(T, leave=False, disable=not progress)):
            phi_i = phi(psi, t)
            re_embeddings[:, s_idx*len_T + t_idx] = np.real(phi_i)
            im_embeddings[:, s_idx*len_T + t_idx] = np.imag(phi_i)

    return np.concatenate([re_embeddings, im_embeddings], axis=1)


if __name__ == '__main__':
    # Demo with karate club graph
    G = nx.karate_club_graph().to_directed()
    N = len(G.nodes)
    q = 0.02
    
    S = HERMLAP_S
    T = HERMLAP_T
   
    embeddings = get_embeddings(get_adj(G), S=S, T=T, q=q, kernel=low_pass_filter_kernel, c=2)
    print(embeddings.shape)
    print(np.count_nonzero(embeddings), embeddings.size)
