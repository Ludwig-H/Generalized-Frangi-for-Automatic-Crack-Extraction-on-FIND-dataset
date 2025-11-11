
import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra

def mst_on_cluster(D: csr_matrix, cluster_idx: np.ndarray) -> csr_matrix:
    sub = D[cluster_idx][:, cluster_idx]
    sub_sym = sub + sub.T
    mst = minimum_spanning_tree(sub_sym)
    mst = mst + mst.T
    return mst.tocsr()

def _farthest_first_kcenters(mst: csr_matrix, k: int) -> List[int]:
    n = mst.shape[0]
    if n == 0: return []
    centers = [0]
    dist = dijkstra(mst, directed=False, indices=centers)[0]
    for _ in range(1, k):
        nxt = int(np.argmax(dist)); centers.append(nxt)
        dn = dijkstra(mst, directed=False, indices=[nxt])[0]
        dist = np.minimum(dist, dn)
    return centers

def kcenters_on_tree(mst: csr_matrix, k: int, objective: str = "max") -> List[int]:
    k = max(1, int(k)); n = mst.shape[0]
    if k >= n: return list(range(n))
    return _farthest_first_kcenters(mst, k)

def _path_nodes_from_predecessors(pred: np.ndarray, src: int, dst: int) -> List[int]:
    path = [dst]; u = dst
    while u != src and u != -9999:
        u = pred[u]
        if u < 0: break
        path.append(u)
    if path[-1] != src: return []
    return path[::-1]

def paths_between_centers(mst: csr_matrix, centers: List[int]) -> Dict[Tuple[int,int], List[int]]:
    paths = {}
    for i, src in enumerate(centers):
        dist, pred = dijkstra(mst, directed=False, return_predecessors=True, indices=src)
        for dst in centers[i+1:]:
            p = _path_nodes_from_predecessors(pred, src, dst)
            if p: paths[(src,dst)] = p
    return paths

def skeleton_from_center_paths(paths: Dict[Tuple[int,int], List[int]],
                               coords: np.ndarray, mst: csr_matrix) -> np.ndarray:
    segs = []
    for (u,v), path in paths.items():
        if len(path)<2: continue
        for a,b in zip(path[:-1], path[1:]):
            w = float(mst[a,b]) if mst[a,b]!=0 else float(mst[b,a])
            r0,c0 = coords[a]; r1,c1 = coords[b]
            segs.append([int(r0),int(c0),int(r1),int(c1),float(w)])
    if len(segs)==0: return np.zeros((0,5), dtype=np.float32)
    return np.array(segs, dtype=np.float32)

def fault_graph_from_mst_and_kcenters(mst: csr_matrix, centers: List[int], weight_agg: str = "mean") -> csr_matrix:
    from scipy.sparse import coo_matrix
    n = mst.shape[0]
    if len(centers)<=1: return coo_matrix((n,n)).tocsr()
    rows, cols, data = [],[],[]
    for i, src in enumerate(centers):
        dist, pred = dijkstra(mst, directed=False, return_predecessors=True, indices=src)
        for dst in centers[i+1:]:
            path = _path_nodes_from_predecessors(pred, src, dst)
            if not path or len(path)<2: continue
            wts=[]
            for a,b in zip(path[:-1], path[1:]):
                w = float(mst[a,b]) if mst[a,b]!=0 else float(mst[b,a]); wts.append(w)
            val = np.median(wts) if weight_agg=="median" else np.mean(wts)
            rows.append(src); cols.append(dst); data.append(val)
    from scipy.sparse import csr_matrix
    G = csr_matrix((data,(rows,cols)), shape=(n,n)); G = G + G.T
    return G
