import time
import numpy as np
import cv2

num_triangles = 50000000 # 50 Million!

u_v = np.random.randint(0, 1000, num_triangles)
v_v = np.random.randint(0, 1000, num_triangles)
w_v = np.random.randint(0, 1000, num_triangles)

id_uv_n = np.random.randint(0, 1000, num_triangles)
id_vw_n = np.random.randint(0, 1000, num_triangles)
id_uw_n = np.random.randint(0, 1000, num_triangles)

e_remap_np = np.arange(1000)
is_valid_node = np.ones(1000, dtype=bool)

# Only 10 edges have centrality > 0 (thin skeleton)
global_dual_cent = np.zeros(1000)
global_dual_cent[np.random.randint(0, 1000, 10)] = np.random.rand(10)

coords_v = np.random.randint(0, 1000, (1000, 2))

cent_img = np.zeros((1000, 1000), dtype=np.float32)
comp_mask = np.zeros((1000, 1000), dtype=np.float32)

t0 = time.time()

# Vectorized extraction
idx_uv = e_remap_np[id_uv_n]
idx_vw = e_remap_np[id_vw_n]
idx_uw = e_remap_np[id_uw_n]

valid_mask = (idx_uv >= 0) & is_valid_node[idx_uv]

idx_uv_v = idx_uv[valid_mask]
idx_vw_v = idx_vw[valid_mask]
idx_uw_v = idx_uw[valid_mask]

val1 = global_dual_cent[idx_uv_v]
val2 = global_dual_cent[idx_vw_v]
val3 = global_dual_cent[idx_uw_v]
vals = np.maximum(np.maximum(val1, val2), val3)

# Filter val > 0
draw_mask = vals > 0
vals_draw = vals[draw_mask]
u_v_draw = u_v[valid_mask][draw_mask]
v_v_draw = v_v[valid_mask][draw_mask]
w_v_draw = w_v[valid_mask][draw_mask]

# Create pts
pts = np.empty((len(u_v_draw), 3, 2), dtype=np.int32)
pts[:, 0, 0] = coords_v[u_v_draw, 1]
pts[:, 0, 1] = coords_v[u_v_draw, 0]
pts[:, 1, 0] = coords_v[v_v_draw, 1]
pts[:, 1, 1] = coords_v[v_v_draw, 0]
pts[:, 2, 0] = coords_v[w_v_draw, 1]
pts[:, 2, 1] = coords_v[w_v_draw, 0]

t1 = time.time()
print(f"Vectorized filter: {t1-t0:.2f}s, Drawing {len(pts)} triangles")

quantized_vals = np.round(vals_draw * 255).astype(np.int32)
unique_bins = np.unique(quantized_vals)
unique_bins = unique_bins[unique_bins > 0]

for b in unique_bins:
    bin_pts = pts[quantized_vals == b]
    for p in bin_pts:
        cv2.fillConvexPoly(cent_img, p, float(b) / 255.0)

for p in pts:
    cv2.fillConvexPoly(comp_mask, p, 1.0)
t2 = time.time()
print(f"Draw time: {t2-t1:.2f}s")
