import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, breadth_first_order

def mst_on_cluster(D: csr_matrix, cluster_idx: np.ndarray) -> csr_matrix:
    sub = D[cluster_idx][:, cluster_idx]
    # D is already symmetric from frangi_graph construction, so sub is symmetric.
    # We essentially need the undirected graph weights.
    mst = minimum_spanning_tree(sub)
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

def extract_backbone_centrality(mst_matrix: csr_matrix, f_threshold: float = 0.1,
                                S: Optional[csr_matrix] = None, take_similarity: bool = True,
                                f_dynamic: bool = False) -> Tuple[np.ndarray, csr_matrix]:
    """
    Extrait le backbone d'un MST en utilisant une chute de centralité relative.
   
    Args:
        mst_matrix (csr_matrix): Le MST (symétrique ou triangulaire).
        f_threshold (float): Facteur de conservation (0.0 à 1.0).
                             Si C_enfant < f * C_parent, la branche est coupée.
        S (csr_matrix): Matrice de similarité (optionnelle).
        take_similarity (bool): Si True, pondère la betweenness par la similarité de l'arête.
        f_dynamic (bool): Si True, le seuil s'adapte au degré du noeud parent pour favoriser les jonctions.
                          Seuil effectif = f_threshold / (degre - 2) si degre > 2.
                             
    Returns:
        backbone_nodes (np.array): Indices des noeuds conservés.
        skeleton_graph (csr_matrix): Le sous-graphe correspondant au backbone.
    """
    N = mst_matrix.shape[0]
    if N == 0:
         return np.array([]), csr_matrix((0,0))
   
    # --- Étape 1 : Calcul Rapide de la Centralité (O(N)) ---
   
    # On enracine arbitrairement en 0 pour orienter le calcul des sous-arbres
    # breadth_first_order renvoie l'ordre de visite et les prédécesseurs
    order, predecessors = breadth_first_order(mst_matrix, i_start=0, directed=False, return_predecessors=True)
   
    # 'predecessors' contient l'index du parent pour chaque noeud (-9999 pour la racine)
    
    # Init poids des noeuds
    node_weights = np.ones(N, dtype=np.float64)
    if take_similarity and S is not None:
        # Poids = Max de la similarité incidente (indépendant de l'arbre)
        node_weights = S.max(axis=1).toarray().flatten().astype(np.float64)
    
    # Calcul des masses de sous-arbres (bottom-up)
    subtree_mass = node_weights.copy()
   
    for i in order[::-1]:
        if i != 0: # Si ce n'est pas la racine de parcours
            parent = predecessors[i]
            # Safety check
            if parent >= 0 and parent < N:
                 subtree_mass[parent] += subtree_mass[i]
           
    # Calcul de la Betweenness Centrality (BC)
    total_mass = subtree_mass[0]
    centrality = subtree_mass * (total_mass - subtree_mass)
   
    # --- Étape 2 : Identifier la vraie racine et filtrer (O(N)) ---
   
    # La "vraie" racine du squelette est le pixel avec la centralité max
    real_root = int(np.argmax(centrality))
   
    # On relance un parcours (BFS) DEPUIS cette vraie racine pour suivre la décroissance
    new_order, new_preds = breadth_first_order(mst_matrix, i_start=real_root, directed=False, return_predecessors=True)
    new_preds[real_root] = real_root # Fix racine

    # --- Étape 2b : Re-calcul de la Centralité sur l'arbre ré-orienté ---
    # C'est CRITIQUE : la centralité précédente était valide pour l'arbre enraciné en 0.
    # Maintenant que la hiérarchie a changé (parent/enfant), il faut recalculer les masses
    # pour que C(v) corresponde bien à la coupure définie par l'arête (parent->v).
    
    subtree_mass = node_weights.copy()
    for i in new_order[::-1]:
        if i != real_root:
            parent = new_preds[i]
            if parent >= 0 and parent < N:
                subtree_mass[parent] += subtree_mass[i]
    
    total_mass = subtree_mass[real_root]
    centrality = subtree_mass * (total_mass - subtree_mass)
   
    # Pré-calcul des degrés pour le mode dynamique
    if f_dynamic:
        # mst_matrix est symétrique, indptr donne le nombre de voisins
        degrees = np.diff(mst_matrix.indptr)

    # Masque des noeuds à garder
    keep_mask = np.zeros(N, dtype=bool)
    keep_mask[real_root] = True
   
    # Parcours topologique (de la racine vers les feuilles)
    for i in new_order:
        if i == real_root:
            continue
           
        parent = new_preds[i]
       
        if keep_mask[parent]:
            current_f = f_threshold
            if f_dynamic:
                d = degrees[parent]
                if d >= 2:
                    current_f = f_threshold / (d - 1.0)
            
            if centrality[i] >= current_f * centrality[parent]:
                keep_mask[i] = True
               
    # --- Étape 3 : Reconstruction du Graphe ---
   
    nodes_to_keep = np.where(keep_mask)[0]
   
    # Extraction du sous-graphe
    skeleton_graph = mst_matrix[nodes_to_keep, :][:, nodes_to_keep]
   
    return nodes_to_keep, skeleton_graph

def skeleton_from_mst_graph(mst_graph: csr_matrix, original_coords: np.ndarray, original_indices: np.ndarray,
                            S: Optional[csr_matrix] = None, take_similarity: bool = True) -> np.ndarray:
    """
    Converts a skeleton graph (subset of MST) into segment list [r0, c0, r1, c1, w].
    
    Args:
        mst_graph: Adjacency matrix of the skeleton (subset of original MST).
        original_coords: Coordinates of ALL nodes in the original cluster (N, 2).
        original_indices: Indices of the kept nodes in the original cluster (M,).
        S: Similarity matrix (original cluster scope).
        take_similarity: If True and S provided, use S values as weights.
        
    Returns:
        segs: (K, 5) array of edges.
    """
    # mst_graph corresponds to nodes indexed 0..M-1. 
    # Node k in mst_graph corresponds to original_indices[k] in the cluster.
    # The coordinate is original_coords[original_indices[k]].
    
    mst_coo = mst_graph.tocoo()
    segs = []
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if u < v: # process each edge once
            orig_u = original_indices[u]
            orig_v = original_indices[v]
            
            r0, c0 = original_coords[orig_u]
            r1, c1 = original_coords[orig_v]
            
            final_w = float(w)
            if take_similarity and S is not None:
                 final_w = float(S[orig_u, orig_v])

            segs.append([float(r0), float(c0), float(r1), float(c1), final_w])
            
    if len(segs) == 0:
        return np.zeros((0,5), dtype=np.float32)
    return np.array(segs, dtype=np.float32)
