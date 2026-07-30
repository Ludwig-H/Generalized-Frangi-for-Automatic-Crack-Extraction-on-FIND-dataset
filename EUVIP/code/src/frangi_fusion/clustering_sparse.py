
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def _symmetrize_min_csr(D: csr_matrix) -> csr_matrix:
    D = D.tocsr(); DT = D.T.tocsr()
    A = D.minimum(DT); B = D.maximum(DT)
    A = A + (B - A).maximum(0); A.eliminate_zeros()
    return A.tocsr()

def _kth_smallest_positive(values: np.ndarray, k: int) -> float:
    vals = values[values>0]
    if vals.size==0: return np.inf
    if vals.size<k: return float(np.max(vals))
    return float(np.partition(vals, k-1)[k-1])

def _core_distances_from_csr(D: csr_matrix, k: int) -> np.ndarray:
    n = D.shape[0]
    core = np.empty(n, dtype=np.float64)
    indptr, data = D.indptr, D.data
    for i in range(n):
        row = data[indptr[i]:indptr[i+1]]
        core[i] = _kth_smallest_positive(row, k) if k>0 else 0.0
    return core

def _mutual_reachability_csr(D: csr_matrix, core: np.ndarray) -> csr_matrix:
    D = D.tocsr()
    indptr, indices, data = D.indptr, D.indices, D.data
    out = np.empty_like(data, dtype=np.float64)
    for i in range(D.shape[0]):
        s,e = indptr[i], indptr[i+1]
        js = indices[s:e]; dij = data[s:e]
        out[s:e] = np.maximum(np.maximum(core[i], core[js]), dij)
    MR = csr_matrix((out, indices.copy(), indptr.copy()), shape=D.shape)
    MR.eliminate_zeros()
    return MR

def _mst_edges_from_sparse(MR: csr_matrix):
    MST = minimum_spanning_tree(MR)
    coo = MST.tocoo()
    edges = {(int(i),int(j)): float(w) for i,j,w in zip(coo.row, coo.col, coo.data)}
    undirected = []
    for (i,j),w in edges.items():
        if (j,i) in edges: w = min(w, edges[(j,i)])
        undirected.append((i,j,w))
    undirected.sort(key=lambda t: t[2])
    return undirected

@dataclass
class ClusterNode:
    id: int
    parent: int
    size: int
    birth_lambda: float
    last_lambda: float
    stability: float
    members: set

class UnionFind:
    def __init__(self, n:int):
        self.parent = np.arange(n, dtype=np.int64)
        self.sz = np.ones(n, dtype=np.int64)
    def find(self, a:int)->int:
        p=self.parent
        while p[a]!=a:
            p[a]=p[p[a]]; a=p[a]
        return a
    def union(self, a:int,b:int)->int:
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return ra
        if self.sz[ra]<self.sz[rb]: ra,rb=rb,ra
        self.parent[rb]=ra; self.sz[ra]+=self.sz[rb]
        return ra

def _build_tree_single_linkage(n:int, edges:List[Tuple[int,int,float]]):
    nodes=[]; uf=UnionFind(n); root2cid={i:i for i in range(n)}
    for i in range(n):
        nodes.append(ClusterNode(id=i,parent=-1,size=1,
                                 birth_lambda=np.inf,last_lambda=np.inf,
                                 stability=0.0,members={i}))
    next_cid=n
    for i,j,w in edges:
        if w<=0: w=1e-12
        lam=1.0/float(w)
        ri,rj=uf.find(i),uf.find(j)
        if ri==rj: continue
        ci,cj=root2cid[ri],root2cid[rj]
        ni,nj=nodes[ci],nodes[cj]
        ni.stability += ni.size*(ni.last_lambda - lam); ni.last_lambda=lam
        nj.stability += nj.size*(nj.last_lambda - lam); nj.last_lambda=lam
        parent_cid=next_cid; next_cid+=1
        members = ni.members | nj.members
        nodes.append(ClusterNode(id=parent_cid,parent=-1,size=ni.size+nj.size,
                                 birth_lambda=lam,last_lambda=lam,stability=0.0,
                                 members=members))
        nodes[ci].parent=parent_cid; nodes[cj].parent=parent_cid
        new_root=uf.union(ri,rj); root2cid[new_root]=parent_cid
    # close to lambda=0
    roots=[nd.id for nd in nodes if nd.parent==-1]
    for rid in roots:
        nd=nodes[rid]; nd.stability += nd.size*(nd.last_lambda - 0.0)
    if len(roots)==1:
        root_id=roots[0]
    else:
        root_id=next_cid; next_cid+=1
        members=set().union(*[nodes[r].members for r in roots])
        nodes.append(ClusterNode(id=root_id,parent=-1,size=len(members),
                                 birth_lambda=0.0,last_lambda=0.0,stability=0.0,
                                 members=members))
        for r in roots: nodes[r].parent=root_id
    parent_to_children={}
    for nd in nodes:
        if nd.parent!=-1:
            parent_to_children.setdefault(nd.parent,[]).append(nd.id)
    return nodes, root_id, parent_to_children

def _select_clusters_eom_iter(nodes, root_id, parent_to_children, min_cluster_size, allow_single_cluster):
    stack=[root_id]; seen=set(); post=[]
    while stack:
        u=stack.pop()
        if u in seen: post.append(u); continue
        seen.add(u); stack.append(u)
        for v in parent_to_children.get(u,[]): stack.append(v)
    best_sum={}; pick_children={}
    for u in post:
        nd=nodes[u]; ch=parent_to_children.get(u,[])
        if not ch:
            best_sum[u]= nd.stability if nd.size>=min_cluster_size else 0.0
            pick_children[u]=False
        else:
            s=sum(best_sum.get(v,0.0) for v in ch)
            self_val= nd.stability if nd.size>=min_cluster_size else 0.0
            if s>self_val: best_sum[u]=s; pick_children[u]=True
            else: best_sum[u]=self_val; pick_children[u]=False
    selected=[]
    stack=[root_id]
    while stack:
        u=stack.pop(); nd=nodes[u]; ch=parent_to_children.get(u,[])
        if not ch:
            if nd.size>=min_cluster_size and nd.stability>0: selected.append(u)
            continue
        if pick_children[u]: stack.extend(ch)
        else:
            if nd.size>=min_cluster_size and nd.stability>0: selected.append(u)
    if not allow_single_cluster and root_id in selected:
        try: selected.remove(root_id)
        except ValueError: pass
    return selected

def _assign_labels_from_selection(n:int, nodes, selected):
    labels=-np.ones(n, dtype=np.int32)
    selected_sorted=sorted(selected, key=lambda cid: nodes[cid].size)
    cur=0
    for cid in selected_sorted:
        for p in nodes[cid].members:
            if labels[p]==-1: labels[p]=cur
        cur+=1
    return labels

def hdbscan_from_sparse(D: csr_matrix, min_cluster_size:int=50, min_samples:int=5,
                        allow_single_cluster:bool=True, expZ:float=2.0)->np.ndarray:
    if not isinstance(D, csr_matrix): raise TypeError("D must be csr_matrix")
    if D.shape[0]!=D.shape[1]: raise ValueError("D must be square")
    D = D.astype(np.float64, copy=False)
    if expZ is not None and expZ!=1.0: D.data **= float(expZ)
    D = _symmetrize_min_csr(D)
    n=D.shape[0]
    if n==0 or D.nnz==0: return -np.ones(n, dtype=np.int32)
    k=int(max(1, min_samples))
    core=_core_distances_from_csr(D, k)
    MR=_mutual_reachability_csr(D, core)
    edges=_mst_edges_from_sparse(MR)
    if len(edges)==0: return -np.ones(n, dtype=np.int32)
    nodes, root_id, p2c = _build_tree_single_linkage(n, edges)
    selected = _select_clusters_eom_iter(nodes, root_id, p2c, min_cluster_size, allow_single_cluster)
    labels = _assign_labels_from_selection(n, nodes, selected)
    return labels
