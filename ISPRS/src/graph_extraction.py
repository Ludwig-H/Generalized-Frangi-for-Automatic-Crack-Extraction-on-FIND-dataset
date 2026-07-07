import time
import cv2
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components
from .frangi_hessian import FrangiHessianGPU

def extract_frangi_graph_gpu(imgs_dict, weights, Σ=[5.0], R=3,
                             ss=1.0, si=0.25, sa=0.3, τ=0.18, min_rel_size=120.0, K=1, device='cuda'):
    t0 = time.time()
    
    fh = FrangiHessianGPU(Σ, device=device)
    
    scale_data = []
    max_S_global = None
    H, W = None, None
    
    # 1. Multi-modal Fusion per scale
    for σ in Σ:
        fused_ixx = None
        
        for mod, w in weights.items():
            if w > 0:
                ixx, ixy, iyy = fh.compute_hessian(imgs_dict[mod], σ)
                if H is None:
                    H, W = ixx.shape
                
                trace = ixx + iyy
                disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
                spectral_norm_local = (torch.abs(trace) + disc) / 2.0
                max_norm = torch.max(spectral_norm_local) + 1e-8
                
                if fused_ixx is None:
                    fused_ixx = w * (ixx / max_norm)
                    fused_ixy = w * (ixy / max_norm)
                    fused_iyy = w * (iyy / max_norm)
                else:
                    fused_ixx += w * (ixx / max_norm)
                    fused_ixy += w * (ixy / max_norm)
                    fused_iyy += w * (iyy / max_norm)
                    
        λ1, λ2, θ = fh.compute_eigenvalues_and_vectors(fused_ixx, fused_ixy, fused_iyy)
        
        mask_pos = λ2 > 0
        R_B = torch.zeros_like(λ2)
        R_B[mask_pos] = torch.abs(λ1[mask_pos]) / (λ2[mask_pos] + 1e-8)
        
        S_norm = torch.zeros_like(λ2)
        S_norm[mask_pos] = λ2[mask_pos]
        
        if max_S_global is None: 
            max_S_global = S_norm.clone()
        else: 
            max_S_global = torch.max(max_S_global, S_norm)
            
        scale_data.append((R_B, S_norm, θ, mask_pos))
        
    if device == 'cuda': 
        torch.cuda.synchronize()
    t_hessian = time.time()
        
    # 2. Node Selection
    τ_1 = max_S_global.max() * 0.01 
    candidates_mask = max_S_global > τ_1
    
    coords = torch.nonzero(candidates_mask).float()
    N = coords.shape[0]
    
    if N == 0:
        return np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W)), {}, {'tau_mask': np.zeros((H, W)), 'comp_mask': np.zeros((H, W))}
    
    index_map = torch.full((H, W), -1, dtype=torch.long, device=device)
    index_map[candidates_mask] = torch.arange(N, device=device)
    
    padded_index_map = torch.nn.functional.pad(index_map.unsqueeze(0).unsqueeze(0).float(), (R, R, R, R), value=-1).long()
    
    patches = padded_index_map[0, 0].unfold(0, 2*R+1, 1).unfold(1, 2*R+1, 1)
    
    y_coords, x_coords = torch.nonzero(candidates_mask, as_tuple=True)
    cand_patches = patches[y_coords, x_coords]
    
    dy, dx = torch.meshgrid(torch.arange(-R, R+1, device=device), torch.arange(-R, R+1, device=device), indexing='ij')
    dist_sq = dx**2 + dy**2
    valid_dist_mask = (dist_sq <= R**2) & (dist_sq > 0)
    half_mask = (dy > 0) | ((dy == 0) & (dx > 0))
    valid_mask = valid_dist_mask & half_mask
    precomputed_dist = torch.sqrt(dist_sq[valid_mask].float())
    
    valid_neighbors = cand_patches[:, valid_mask]
    
    source_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, valid_neighbors.shape[1])
    valid_pairs_mask = valid_neighbors != -1
    
    i_idx_t = source_idx[valid_pairs_mask]
    j_idx_t = valid_neighbors[valid_pairs_mask]
    
    if len(i_idx_t) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W)), np.zeros((H, W)), {}, {'tau_mask': np.zeros((H, W)), 'comp_mask': np.zeros((H, W))}
    
    if device == 'cuda': 
        torch.cuda.synchronize()
    t_unfold = time.time()
    
    # 3. Compute similarities and distances
    S_ij_max = torch.zeros(len(i_idx_t), device=device, dtype=torch.float32)
    chunk_size = 5000000
    for chunk_start in range(0, len(i_idx_t), chunk_size):
        chunk_end = min(len(i_idx_t), chunk_start + chunk_size)
        i_chunk = i_idx_t[chunk_start:chunk_end]
        j_chunk = j_idx_t[chunk_start:chunk_end]
        
        S_ij_chunk_max = torch.zeros(chunk_end - chunk_start, device=device, dtype=torch.float32)
        
        for R_B, S_norm, θ, mask_pos in scale_data:
            rb_c = R_B[candidates_mask]
            s_c  = S_norm[candidates_mask]
            t_c  = θ[candidates_mask]
            
            rb_sum = rb_c[i_chunk] + rb_c[j_chunk]
            S_shape = torch.exp(-0.5 * (rb_sum / ss)**2)
            
            s_prod = s_c[i_chunk] * s_c[j_chunk]
            S_int = 1 - torch.exp(-0.5 * (s_prod / si)**2)
            
            dt = t_c[i_chunk] - t_c[j_chunk]
            S_align = torch.exp(-0.5 * (torch.sin(dt) / sa)**2)
            
            S_ij = S_shape * S_int * S_align
            S_ij_chunk_max = torch.max(S_ij_chunk_max, S_ij)
            
        S_ij_max[chunk_start:chunk_end] = S_ij_chunk_max
        
    if device == 'cuda': 
        torch.cuda.synchronize()
    t_sim = time.time()
        
    dist_matrix = precomputed_dist.unsqueeze(0).expand(N, -1)
    dist_ij_t = dist_matrix[valid_pairs_mask]
    d_ij = (1 - S_ij_max) * dist_ij_t + 1e-8
    
    S_cpu = S_ij_max.cpu().numpy()
    d_cpu = d_ij.cpu().numpy()
    i_cpu = i_idx_t.cpu().numpy()
    j_cpu = j_idx_t.cpu().numpy()
    
    num_edges = len(S_ij_max)
    num_to_keep_edges = max(1, int(num_edges * τ))
    
    if num_edges > num_to_keep_edges:
        threshold_edge = torch.kthvalue(S_ij_max, num_edges - num_to_keep_edges + 1).values.item()
        edge_mask_t = S_ij_max >= threshold_edge
        edge_mask = edge_mask_t.cpu().numpy()
    else:
        edge_mask = np.ones(num_edges, dtype=bool)
    
    i_v = i_cpu[edge_mask]
    j_v = j_cpu[edge_mask]
    S_v = S_cpu[edge_mask]
    d_v = d_cpu[edge_mask]
    
    node_sim_max = np.zeros(N, dtype=np.float32)
    if len(S_v) > 0:
        np.maximum.at(node_sim_max, i_v, S_v)
        np.maximum.at(node_sim_max, j_v, S_v)
        
    N_total = H * W
    num_to_keep_nodes = max(1, int(N_total * τ))
    
    valid_nodes_candidates = np.unique(np.concatenate([i_v, j_v])) if len(i_v) > 0 else np.array([], dtype=np.int32)
    
    if len(valid_nodes_candidates) > num_to_keep_nodes:
        sims_candidates = node_sim_max[valid_nodes_candidates]
        threshold_node = np.partition(sims_candidates, -num_to_keep_nodes)[-num_to_keep_nodes]
        valid_nodes = valid_nodes_candidates[sims_candidates >= threshold_node]
    else:
        valid_nodes = valid_nodes_candidates
        
    if len(valid_nodes) > 0 and len(i_v) > 0:
        keep_node_mask = np.zeros(N, dtype=bool)
        keep_node_mask[valid_nodes] = True
        
        final_edge_mask = keep_node_mask[i_v] & keep_node_mask[j_v]
        i_v = i_v[final_edge_mask]
        j_v = j_v[final_edge_mask]
        S_v = S_v[final_edge_mask]
        d_v = d_v[final_edge_mask]
    else:
        i_v, j_v, S_v, d_v = np.array([]), np.array([]), np.array([]), np.array([])
    
    remap = np.full(N, -1, dtype=np.int32)
    remap[valid_nodes] = np.arange(len(valid_nodes))
    
    i_mapped = remap[i_v]
    j_mapped = remap[j_v]
    
    orig_coords_cpu = coords.cpu().numpy().astype(int)
    valid_nodes_t = torch.from_numpy(valid_nodes).to(device).long()
    coords = coords[valid_nodes_t]
    N_valid = len(valid_nodes)
    
    cent_img = np.zeros((H, W), dtype=np.float32)
    comp_mask = np.zeros((H, W), dtype=np.float32)
    tau_mask = np.zeros((H, W), dtype=np.float32)
    tau_mask[orig_coords_cpu[valid_nodes, 0], orig_coords_cpu[valid_nodes, 1]] = 1.0
    
    t_mst_total = 0
    t_bet_total = 0

    if K == 1:
        sparse_dist = coo_matrix((np.concatenate([d_v, d_v]), 
                                  (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                                 shape=(N_valid, N_valid)).tocsr()
        sparse_sim = coo_matrix((np.concatenate([S_v, S_v]), 
                                 (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                                shape=(N_valid, N_valid)).tocsr()

        n_comp, labels = connected_components(sparse_dist, directed=False)
        counts = np.bincount(labels)
        min_size = N_total / min_rel_size
        valid_components = np.where(counts > min_size)[0]
        
        if len(valid_components) > 0:
            t_bet_start = time.time()
            for comp_id in valid_components:
                mask_comp = (labels == comp_id)
                nodes_comp = np.where(mask_comp)[0]
                
                sparse_dist_comp = sparse_dist[nodes_comp, :][:, nodes_comp]
                S_sparse_comp = sparse_sim[nodes_comp, :][:, nodes_comp]
                
                t_mst_start = time.time()
                mst = minimum_spanning_tree(sparse_dist_comp)
                order, preds = breadth_first_order(mst, i_start=0, directed=False, return_predecessors=True)
                t_mst_total += (time.time() - t_mst_start)
                
                N_L = len(nodes_comp)
                valid_mask_preds = preds >= 0
                p_valid, i_valid = preds[valid_mask_preds], np.arange(N_L)[valid_mask_preds]
                
                W_parent_np = np.zeros(N_L, dtype=np.float32)
                if len(p_valid) > 0:
                    w1 = np.asarray(S_sparse_comp[p_valid, i_valid]).flatten()
                    w2 = np.asarray(S_sparse_comp[i_valid, p_valid]).flatten()
                    W_parent_np[i_valid] = np.maximum(w1, w2)

                E_mass_np = np.zeros(N_L, dtype=np.float32)
                for i in order[::-1]:
                    p = preds[i]
                    if p >= 0: 
                        E_mass_np[p] += E_mass_np[i] + W_parent_np[i]
                        
                M_total = float(E_mass_np[order[0]])
                W_p = torch.tensor(W_parent_np, dtype=torch.float32, device=device)
                E_m = torch.tensor(E_mass_np, dtype=torch.float32, device=device)
                
                M_child = E_m + W_p
                
                sum_M = torch.zeros(N_L, dtype=torch.float32, device=device)
                sum_M2 = torch.zeros(N_L, dtype=torch.float32, device=device)
                p_v_t = torch.tensor(p_valid, dtype=torch.long, device=device)
                i_v_t = torch.tensor(i_valid, dtype=torch.long, device=device)
                
                sum_M.index_add_(0, p_v_t, M_child[i_v_t])
                sum_M2.index_add_(0, p_v_t, M_child[i_v_t]**2)
                
                C_children = 0.5 * (sum_M**2 - sum_M2)
                M_parent_branch = torch.clamp(M_total - sum_M, min=0.0)
                
                centrality = C_children + sum_M * M_parent_branch
                if centrality.max() > 0: 
                    centrality /= centrality.max()
                    
                coords_v = coords.cpu().numpy()
                u_m, v_m = mst.nonzero()
                for i_e in range(len(u_m)):
                    idx1, idx2 = u_m[i_e], v_m[i_e]
                    n1, n2 = nodes_comp[idx1], nodes_comp[idx2]
                    c1, c2 = coords_v[n1], coords_v[n2]
                    val = max(centrality[idx1], centrality[idx2])
                    cv2.line(cent_img, (int(c1[1]), int(c1[0])), (int(c2[1]), int(c2[0])), float(val), 1)
                    cv2.line(comp_mask, (int(c1[1]), int(c1[0])), (int(c2[1]), int(c2[0])), 1.0, 1)

            if device == 'cuda': 
                torch.cuda.synchronize()
            t_bet_total = time.time() - t_bet_start - t_mst_total

    elif K == 2:
        adj_i = i_mapped
        adj_j = j_mapped
        num_e_init = len(adj_i)
        
        if num_e_init >= 3:
            adj_i_t = torch.from_numpy(adj_i).to(device).long()
            adj_j_t = torch.from_numpy(adj_j).to(device).long()
            
            u_t = torch.minimum(adj_i_t, adj_j_t)
            v_t = torch.maximum(adj_i_t, adj_j_t)
            
            u_sym = torch.cat([u_t, v_t])
            v_sym = torch.cat([v_t, u_t])
            
            sort_idx = torch.argsort(u_sym)
            u_sym = u_sym[sort_idx]
            v_sym = v_sym[sort_idx]
            
            deg = torch.bincount(u_sym, minlength=N_valid)
            max_deg = deg.max().item()
            
            indptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(deg, dim=0)])
            
            row_ids = torch.arange(N_valid, device=device).repeat_interleave(deg)
            col_ids = torch.arange(len(v_sym), device=device) - indptr[row_ids]
            
            neigh_matrix = torch.full((N_valid, max_deg), -1, dtype=torch.long, device=device)
            neigh_matrix[row_ids, col_ids] = v_sym
            
            neigh_matrix, _ = torch.sort(neigh_matrix, dim=1)
            
            tri_u, tri_v, tri_w = [], [], []
            E_total = len(u_t)
            batch_size = 100000
            
            for i in range(0, E_total, batch_size):
                end_i = min(E_total, i + batch_size)
                u_b = u_t[i:end_i]
                w_b = v_t[i:end_i]
                
                n_u = neigh_matrix[u_b]
                n_w = neigh_matrix[w_b]
                
                idx = torch.searchsorted(n_w, n_u)
                idx = idx.clamp(max=max(0, max_deg - 1))
                v_vals_candidate = torch.gather(n_w, 1, idx)
                match = (v_vals_candidate == n_u) & (n_u != -1)
                
                b_idx, r_idx = torch.where(match)
                if len(b_idx) > 0:
                    v_vals = n_u[b_idx, r_idx]
                    u_vals = u_b[b_idx]
                    w_vals = w_b[b_idx]
                    
                    valid_tri = (u_vals < v_vals) & (v_vals < w_vals)
                    
                    if valid_tri.any():
                        tri_u.append(u_vals[valid_tri].cpu().numpy())
                        tri_v.append(v_vals[valid_tri].cpu().numpy())
                        tri_w.append(w_vals[valid_tri].cpu().numpy())

            if len(tri_u) > 0:
                tri_u = np.concatenate(tri_u)
                tri_v = np.concatenate(tri_v)
                tri_w = np.concatenate(tri_w)
                ids = np.arange(num_e_init, dtype=np.int32) + 1
                
                u_l_np = u_t.cpu().numpy()
                v_l_np = v_t.cpu().numpy()
                
                edge_to_id_sparse = sp.csr_matrix(
                    (np.concatenate([ids, ids]), 
                     (np.concatenate([u_l_np, v_l_np]), np.concatenate([v_l_np, u_l_np]))), 
                    shape=(N_valid, N_valid)
                )
                
                u_tri_np, v_tri_np, w_tri_np = np.array(tri_u), np.array(tri_v), np.array(tri_w)
                
                id_uv = np.asarray(edge_to_id_sparse[u_tri_np, v_tri_np]).flatten() - 1
                id_vw = np.asarray(edge_to_id_sparse[v_tri_np, w_tri_np]).flatten() - 1
                id_uw = np.asarray(edge_to_id_sparse[u_tri_np, w_tri_np]).flatten() - 1
                
                valid_tri = (id_uv >= 0) & (id_vw >= 0) & (id_uw >= 0)
                
                if valid_tri.any():
                    u_tri = torch.from_numpy(u_tri_np[valid_tri]).to(device).long()
                    v_tri = torch.from_numpy(v_tri_np[valid_tri]).to(device).long()
                    w_tri = torch.from_numpy(w_tri_np[valid_tri]).to(device).long()
                    
                    id_uv = torch.from_numpy(id_uv[valid_tri]).to(device).long()
                    id_vw = torch.from_numpy(id_vw[valid_tri]).to(device).long()
                    id_uw = torch.from_numpy(id_uw[valid_tri]).to(device).long()
                    
                    adj_i_t = torch.from_numpy(adj_i).to(device).long()
                    adj_j_t = torch.from_numpy(adj_j).to(device).long()
                    
                    d_v_t = torch.from_numpy(d_v).to(device)
                    S_v_t = torch.from_numpy(S_v).to(device)
                    D_T = torch.max(torch.stack([d_v_t[id_uv], d_v_t[id_vw], d_v_t[id_uw]]), dim=0).values
                    S_T = torch.min(torch.stack([S_v_t[id_uv], S_v_t[id_vw], S_v_t[id_uw]]), dim=0).values
                    
                    active_e = torch.unique(torch.cat([id_uv, id_vw, id_uw]))
                    num_act_e = len(active_e)
                    e_remap = torch.full((len(adj_i_t),), -1, dtype=torch.long, device=device)
                    e_remap[active_e] = torch.arange(num_act_e, device=device)
                    
                    di = torch.cat([e_remap[id_uv], e_remap[id_vw], e_remap[id_uw]])
                    dj = torch.cat([e_remap[id_vw], e_remap[id_uw], e_remap[id_uv]])
                    dw = torch.cat([D_T, D_T, D_T])
                    ds = torch.cat([S_T, S_T, S_T])
                    
                    di_c, dj_c = di.cpu().numpy(), dj.cpu().numpy()
                    dw_c, ds_c = dw.cpu().numpy(), ds.cpu().numpy()
                    
                    sparse_dual = coo_matrix((dw_c, (di_c, dj_c)), shape=(num_act_e, num_act_e)).tocsr()
                    sparse_dual_S = coo_matrix((ds_c, (di_c, dj_c)), shape=(num_act_e, num_act_e)).tocsr()
                    
                    n_comp, labels = connected_components(sparse_dual, directed=False)
                    counts = np.bincount(labels)
                    
                    adj_i_v, adj_j_v = adj_i_t.cpu().numpy(), adj_j_t.cpu().numpy()
                    
                    v_comp = []
                    for c_id in range(n_comp):
                        if counts[c_id] == 0: 
                            continue
                        
                        m_comp = (labels == c_id)
                        n_comp_idx = np.where(m_comp)[0]
                        
                        e_indices = active_e[n_comp_idx].cpu().numpy()
                        u_nodes = adj_i_v[e_indices]
                        v_nodes = adj_j_v[e_indices]
                        unique_nodes = np.unique(np.concatenate([u_nodes, v_nodes]))
                        
                        if len(unique_nodes) > (N_total / min_rel_size):
                            v_comp.append(c_id)
                    
                    global_dual_cent = np.zeros(num_act_e, dtype=np.float32)
                    is_valid_node = np.zeros(num_act_e, dtype=bool)
                    coords_v = coords.cpu().numpy()
                    t_bet_start = time.time()
                    
                    for c_id in v_comp:
                        m_comp = (labels == c_id)
                        n_comp_idx = np.where(m_comp)[0]
                        s_dual_dist_c = sparse_dual[n_comp_idx, :][:, n_comp_idx]
                        s_dual_S_c = sparse_dual_S[n_comp_idx, :][:, n_comp_idx]
                        
                        t_mst_s = time.time()
                        mst = minimum_spanning_tree(s_dual_dist_c)
                        order, preds = breadth_first_order(mst, i_start=0, directed=False, return_predecessors=True)
                        t_mst_total += (time.time() - t_mst_s)
                        
                        N_L = len(n_comp_idx)
                        v_m_preds = preds >= 0
                        p_v, i_v_l = preds[v_m_preds], np.arange(N_L)[v_m_preds]
                        
                        W_p_np = np.zeros(N_L, dtype=np.float32)
                        if len(p_v) > 0:
                            w1 = np.asarray(s_dual_S_c[p_v, i_v_l]).flatten()
                            w2 = np.asarray(s_dual_S_c[i_v_l, p_v]).flatten()
                            W_p_np[i_v_l] = np.maximum(w1, w2)
                        
                        E_m_np = np.zeros(N_L, dtype=np.float32)
                        for i in order[::-1]:
                            p = preds[i]
                            if p >= 0: 
                                E_m_np[p] += E_m_np[i] + W_p_np[i]
                            
                        M_tot = float(E_m_np[order[0]])
                        W_p = torch.tensor(W_p_np, dtype=torch.float32, device=device)
                        E_m = torch.tensor(E_m_np, dtype=torch.float32, device=device)
                        
                        M_child_dual = E_m + W_p
                        
                        sum_M_dual = torch.zeros(N_L, dtype=torch.float32, device=device)
                        sum_M2_dual = torch.zeros(N_L, dtype=torch.float32, device=device)
                        p_v_t, i_v_t = torch.tensor(p_v, dtype=torch.long, device=device), torch.tensor(i_v_l, dtype=torch.long, device=device)
                        
                        sum_M_dual.index_add_(0, p_v_t, M_child_dual[i_v_t])
                        sum_M2_dual.index_add_(0, p_v_t, M_child_dual[i_v_t]**2)
                        
                        C_child_dual = 0.5 * (sum_M_dual**2 - sum_M2_dual)
                        M_p_dual = torch.clamp(M_tot - sum_M_dual, min=0.0)
                        
                        centrality = C_child_dual + sum_M_dual * M_p_dual
                        if centrality.max() > 0: 
                            centrality /= centrality.max()
                        
                        global_dual_cent[n_comp_idx] = centrality.cpu().numpy()
                        is_valid_node[n_comp_idx] = True
 
                    u_v, v_v, w_v = u_tri.cpu().numpy(), v_tri.cpu().numpy(), w_tri.cpu().numpy()
                    id_uv_n, id_vw_n, id_uw_n = id_uv.cpu().numpy(), id_vw.cpu().numpy(), id_uw.cpu().numpy()
                    e_remap_np = e_remap.cpu().numpy()
                    
                    idx_uv = e_remap_np[id_uv_n]
                    idx_vw = e_remap_np[id_vw_n]
                    idx_uw = e_remap_np[id_uw_n]
                    
                    valid_mask = (idx_uv >= 0) & is_valid_node[idx_uv]
                    
                    if valid_mask.any():
                        idx_uv_v = idx_uv[valid_mask]
                        idx_vw_v = idx_vw[valid_mask]
                        idx_uw_v = idx_uw[valid_mask]
                        
                        val1 = global_dual_cent[idx_uv_v]
                        val2 = global_dual_cent[idx_vw_v]
                        val3 = global_dual_cent[idx_uw_v]
                        vals = np.maximum(np.maximum(val1, val2), val3)
                        
                        draw_mask = vals > 0
                        if draw_mask.any():
                            vals_draw = vals[draw_mask]
                            u_v_draw = u_v[valid_mask][draw_mask]
                            v_v_draw = v_v[valid_mask][draw_mask]
                            w_v_draw = w_v[valid_mask][draw_mask]
                            
                            pts = np.empty((len(u_v_draw), 3, 2), dtype=np.int32)
                            pts[:, 0, 0] = coords_v[u_v_draw, 1]
                            pts[:, 0, 1] = coords_v[u_v_draw, 0]
                            pts[:, 1, 0] = coords_v[v_v_draw, 1]
                            pts[:, 1, 1] = coords_v[v_v_draw, 0]
                            pts[:, 2, 0] = coords_v[w_v_draw, 1]
                            pts[:, 2, 1] = coords_v[w_v_draw, 0]
                            
                            quantized_vals = np.round(vals_draw * 255).astype(np.int32)
                            unique_bins = np.unique(quantized_vals)
                            unique_bins = unique_bins[unique_bins > 0]
                            
                            for b in unique_bins:
                                bin_pts = pts[quantized_vals == b]
                                for p in bin_pts:
                                    cv2.fillConvexPoly(cent_img, p, float(b) / 255.0)
 
                            for p in pts:
                                cv2.fillConvexPoly(comp_mask, p, 1.0)                                    
                                    
                    if device == 'cuda': 
                        torch.cuda.synchronize()
                    t_bet_total = time.time() - t_bet_start - t_mst_total
    
    sim_img = np.zeros((H, W), dtype=np.float32)
    sim_img[orig_coords_cpu[:, 0], orig_coords_cpu[:, 1]] = node_sim_max
    
    if device == 'cuda': 
        torch.cuda.synchronize()
    t_end = time.time()
    
    timings = {
        "1. Hessian Fusion": t_hessian - t0,
        "2. Graph Unfold": t_unfold - t_hessian,
        "3. Frangi Similarity": t_sim - t_unfold,
        "4. MST (CPU)": t_mst_total,
        "5. Betweenness (GPU)": t_bet_total,
        "Total": t_end - t0
    }
    
    return max_S_global.cpu().numpy(), sim_img, cent_img, timings, {'tau_mask': tau_mask, 'comp_mask': comp_mask}
