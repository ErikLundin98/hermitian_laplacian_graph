import networkx as nx
from lib import hermitian_laplacian
G = nx.DiGraph()

G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 0)
G.add_edge(0, 2)

L_q = hermitian_laplacian(G, q=0)

print(L_q)