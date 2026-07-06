import torch
import torch.nn.functional as F
import math

class FrangiHessianGPU:
    def __init__(self, scales, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.scales = scales
        self.device = device
        self.kernels = self._precompute_kernels()

    def _precompute_kernels(self):
        kernels = {}
        for s in self.scales:
            size = int(math.ceil(4 * s)) * 2 + 1
            x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.device)
            variance = s ** 2
            g_x = (1 / (math.sqrt(2 * math.pi) * s)) * torch.exp(-x**2 / (2 * variance))
            g_x_1 = -(x / variance) * g_x
            g_x_2 = ((x**2 / (variance**2)) - (1 / variance)) * g_x
            kernels[s] = {'0': g_x, '1': g_x_1, '2': g_x_2}
        return kernels

    def compute_hessian(self, image_tensor, s):
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            
        image_tensor = image_tensor.to(self.device)
        
        k0 = self.kernels[s]['0'].view(1, 1, 1, -1)
        k1 = self.kernels[s]['1'].view(1, 1, 1, -1)
        k2 = self.kernels[s]['2'].view(1, 1, 1, -1)

        k0_T = k0.transpose(2, 3)
        k1_T = k1.transpose(2, 3)
        k2_T = k2.transpose(2, 3)

        pad_size = k0.shape[3] // 2
        
        def convolve_sep(x, k_h, k_v):
            p_h = F.pad(x, (pad_size, pad_size, 0, 0), mode='replicate')
            t = F.conv2d(p_h, k_h)
            p_v = F.pad(t, (0, 0, pad_size, pad_size), mode='replicate')
            return F.conv2d(p_v, k_v)

        ixx = convolve_sep(image_tensor, k2, k0_T)
        iyy = convolve_sep(image_tensor, k0, k2_T)
        ixy = convolve_sep(image_tensor, k1, k1_T)

        return ixx.squeeze(), ixy.squeeze(), iyy.squeeze()

    def compute_eigenvalues_and_vectors(self, ixx, ixy, iyy):
        trace = ixx + iyy
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        l_plus = (trace + disc) / 2
        l_minus = (trace - disc) / 2
        
        abs_l_plus = torch.abs(l_plus)
        abs_l_minus = torch.abs(l_minus)
        
        mask_minus_bigger = abs_l_minus > abs_l_plus
        
        λ2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        λ1 = torch.where(mask_minus_bigger, l_plus, l_minus)

        θ = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        return λ1, λ2, θ
