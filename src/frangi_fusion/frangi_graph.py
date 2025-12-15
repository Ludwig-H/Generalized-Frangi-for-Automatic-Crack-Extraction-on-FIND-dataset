
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

def _sim_elong(l1a, l2a, l1b, l2b, beta: float):
    ra = np.abs(l1a) / (np.abs(l2a) + 1e-12)
    rb = np.abs(l1b) / (np.abs(l2b) + 1e-12)
    r = ra + rb
    return np.exp(-0.5 * (r/(beta+1e-12))**2)

def _sim_strength(l2a, l2b, c: float):
    prod = np.abs(l2a * l2b)
    return 1.0 - np.exp(-0.5 * (np.sqrt(prod)/(c+1e-12))**2)

def _sim_angle(thet_a, thet_b, ctheta: float):
    diff = np.sin(thet_a - thet_b)
    return np.exp(-0.5 * (np.abs(diff)/(ctheta+1e-12))**2)

def build_frangi_similarity_graph(fused_hessians: List[Dict[str,np.ndarray]],
                                  beta: float, c: float, ctheta: float, R: int,
                                  candidate_mask: Optional[np.ndarray] = None,
                                  threshold_mask: Optional[float] = None,
                                  dark_ridges: bool = True,
                                  take_distances: bool = True
                                  ) -> Tuple[np.ndarray, List[List[int]], csr_matrix]:
    # pick scale-wise maxima by |λ2n|
    e1s = [Hd["e1n"] for Hd in fused_hessians]
    e2s = [Hd["e2n"] for Hd in fused_hessians]
    thetas = [Hd["theta"] for Hd in fused_hessians]
    H, W = e2s[0].shape

    resp = np.max([np.abs(e2) for e2 in e2s], axis=0)
    if candidate_mask is None:
        if threshold_mask is None:
            threshold_mask = 0.5 # CHANGEMENT 0.95
        thr = np.quantile(resp, threshold_mask) 
        candidate_mask = resp >= thr
    coords = np.argwhere(candidate_mask)
    if coords.size == 0:
        idx = np.argsort(resp.reshape(-1))[-500:]
        coords = np.column_stack(np.unravel_index(idx, (H,W)))
    N = coords.shape[0]

    tree = cKDTree(coords[:, ::-1])
    pairs = tree.query_pairs(r=R, output_type='ndarray')
    if pairs.size == 0:
        return coords, [[] for _ in range(N)], csr_matrix((N,N))

    def sim_at_scale(sidx: int, dark_flag: bool) -> np.ndarray:
        e1 = e1s[sidx]; e2 = e2s[sidx]; th = thetas[sidx]
        r0, c0 = coords[pairs[:, 0], 0], coords[pairs[:, 0], 1]
        r1, c1 = coords[pairs[:, 1], 0], coords[pairs[:, 1], 1]
        l1a, l2a = e1[r0, c0], e2[r0, c0]
        l1b, l2b = e1[r1, c1], e2[r1, c1]
        ta, tb = th[r0, c0], th[r1, c1]

        # masque selon le signe choisi
        valid = (l2a >= 0) & (l2b >= 0) if dark_flag else (l2a <= 0) & (l2b <= 0)

        s1 = _sim_elong(l1a, l2a, l1b, l2b, beta)
        s2 = _sim_strength(l2a, l2b, c)
        s3 = _sim_angle(ta, tb, ctheta)
        sims = s1 * s2 * s3
        sims[~valid] = 0.0
        return sims

    threshold_quant_dark = 0.95
    # Passe 1: hypothèse initiale (dark_ridges)
    sims_scales_a = [sim_at_scale(i, dark_ridges) for i in range(len(e2s))]
    sims_a = np.max(np.vstack(sims_scales_a), axis=0)
    q_a = np.quantile(sims_a, threshold_quant_dark)

    # Passe 2: hypothèse contraire
    dark_alt = (not dark_ridges)
    sims_scales_b = [sim_at_scale(i, dark_alt) for i in range(len(e2s))]
    sims_b = np.max(np.vstack(sims_scales_b), axis=0)
    q_b = np.quantile(sims_b, threshold_quant_dark)

    # if q_b > q_a:
    #     sims = sims_b
    #     chosen = dark_alt
    # else:
    #     sims = sims_a
    #     chosen = dark_ridges
    # print(f"Quantiles: {q_a:.6f} / {q_b:.6f}. Chosen dark_ridges = {chosen}")
    if dark_ridges :
        sims = sims_a
    else :
        sims = sims_b

    if take_distances:
        r0 = coords[pairs[:, 0], 0]
        c0 = coords[pairs[:, 0], 1]
        r1 = coords[pairs[:, 1], 0]
        c1 = coords[pairs[:, 1], 1]
        d_ij = np.sqrt((r0 - r1)**2 + (c0 - c1)**2)
        sims = np.maximum(0, 1 - (1 - sims) * d_ij)
                                      
    row = pairs[:,0]; col = pairs[:,1]; data = sims
    S = coo_matrix((data,(row,col)), shape=(N,N))
    S = (S + S.T)/2
    S = S.tocsr()
    neighbors = [[] for _ in range(N)]
    cooS = S.tocoo()
    for i,j,v in zip(cooS.row, cooS.col, cooS.data):
        if i!=j and v>0: neighbors[i].append(j)
    return coords, neighbors, S

def distances_from_similarity(S: csr_matrix, mode="inverse") -> csr_matrix:
    Sd = S.copy().astype(np.float64)
    if mode == "minus" :
        Sd.data = np.clip(1.0 - Sd.data, 0.0, 1.0)
    elif mode == "log" :
        Sd.data = np.sqrt(-np.log(Sd.data))
    else :
        Sd.data = 1 / Sd.data -1
    return Sd

def triangle_connectivity_graph(coords: np.ndarray, D: csr_matrix, max_triangles_per_node: int = 50) -> csr_matrix:
    import itertools
    from collections import defaultdict
    N = D.shape[0]; D = D.tocsr()
    neighbors = [D.indices[D.indptr[i]:D.indptr[i+1]] for i in range(N)]
    weights   = [D.data[D.indptr[i]:D.indptr[i+1]] for i in range(N)]
    def edge_id(i,j): return (i,j) if i<j else (j,i)
    edge_w = {}
    for i in range(N):
        for j,w in zip(neighbors[i], weights[i]):
            if i<j: edge_w[(i,j)] = float(w)
    triangles = []
    for u in range(N):
        nbrs = neighbors[u]
        if len(nbrs)<2: continue
        cnt = 0
        for v,w in itertools.combinations(nbrs,2):
            if v==w or v==u or w==u: continue
            a = edge_id(min(v,w), max(v,w))
            if a not in edge_w: continue
            e_uv = edge_id(u,v); e_uw = edge_id(u,w); e_vw = edge_id(min(v,w), max(v,w))
            if e_uv not in edge_w or e_uw not in edge_w or e_vw not in edge_w: continue
            d_uv = edge_w[e_uv]; d_uw = edge_w[e_uw]; d_vw = edge_w[e_vw]
            tri_val = max(d_uv,d_uw,d_vw)
            e_min = min([(d_uv,e_uv),(d_uw,e_uw),(d_vw,e_vw)], key=lambda x:x[0])[1]
            triangles.append((tri_val,(u,v,w),e_min))
            cnt += 1
            if cnt>=max_triangles_per_node: break
    if not triangles: return D
    from collections import defaultdict
    best_tri_for_node = {}
    for tri_val,(u,v,w),e_min in triangles:
        for x in (u,v,w):
            if x not in best_tri_for_node or tri_val < best_tri_for_node[x][0]:
                best_tri_for_node[x] = (tri_val, e_min)
    edge_to_nodes = defaultdict(set)
    for n,(tri_val,e_min) in best_tri_for_node.items():
        edge_to_nodes[e_min].add(n)
    parent = list(range(N))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def unite(a,b):
        ra,rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    for tri_val,(u,v,w),e_min in triangles:
        e_uv = (min(u,v),max(u,v)); e_uw=(min(u,w),max(u,w)); e_vw=(min(v,w),max(v,w))
        groups = [edge_to_nodes.get(e_uv,set()), edge_to_nodes.get(e_uw,set()), edge_to_nodes.get(e_vw,set())]
        all_nodes = set().union(*groups)
        if not all_nodes: continue
        base = next(iter(all_nodes))
        for x in all_nodes: unite(base,x)
    rows, cols, data = [],[],[]
    for i in range(N):
        ci = find(i)
        for j,w in zip(neighbors[i], weights[i]):
            if i<j and find(j)==ci:
                rows.append(i); cols.append(j); data.append(w)
    M = coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()
    M = M + M.T
    return M
