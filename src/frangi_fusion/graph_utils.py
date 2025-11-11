
import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix, csr_matrix, csgraph

def csr_from_edges(n: int, edges: List[Tuple[int,int,float]]) -> csr_matrix:
    if not edges: return csr_matrix((n,n))
    rows = [i for i,j,w in edges]
    cols = [j for i,j,w in edges]
    data = [w for i,j,w in edges]
    M = coo_matrix((data,(rows,cols)), shape=(n,n)).tocsr()
    M = M + M.T
    return M

def largest_connected_component(M: csr_matrix):
    n = M.shape[0]
    if n==0: return M, np.array([], dtype=int)
    graph = (M > 0).astype(int)
    n_components, labels = csgraph.connected_components(graph, directed=False)
    if n_components <= 1: return M, np.arange(n, dtype=int)
    counts = np.bincount(labels)
    comp_id = counts.argmax()
    idx = np.where(labels==comp_id)[0]
    sub = M[idx][:, idx]
    return sub, idx

def subgraph_by_nodes(M: csr_matrix, idx: np.ndarray) -> csr_matrix:
    return M[idx][:, idx]
