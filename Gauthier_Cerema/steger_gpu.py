import torch
import torch.nn.functional as F
import math

class StegerHessian:
    def __init__(self, σ, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.σ = σ
        self.device = device
        self.kernels = self._precompute_kernels()

    def _precompute_kernels(self):
        """
        Creates Gaussian derivative kernels for convolution.
        Returns a dictionary of kernels for 0th, 1st, and 2nd derivatives.
        """
        # Determine kernel size (usually 3*σ to 6*σ)
        # We use 4*σ to be safe and ensure odd size
        size = int(math.ceil(4 * self.σ)) * 2 + 1
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.device)
        
        # Gaussian function G(x)
        # G(x) = (1 / (sqrt(2pi) * σ)) * exp(-x^2 / 2σ^2)
        variance = self.σ ** 2
        g_x = (1 / (math.sqrt(2 * math.pi) * self.σ)) * torch.exp(-x**2 / (2 * variance))
        
        # First derivative G'(x) = -x/σ^2 * G(x)
        g_x_1 = -(x / variance) * g_x
        
        # Second derivative G''(x) = (x^2/σ^4 - 1/σ^2) * G(x)
        g_x_2 = ((x**2 / (variance**2)) - (1 / variance)) * g_x

        return {
            '0': g_x,
            '1': g_x_1,
            '2': g_x_2
        }

    def compute_hessian(self, image_tensor):
        """
        Computes Hessian components (Ixx, Ixy, Iyy) for a batch of images.
        Uses replication padding to minimize boundary effects.
        image_tensor: (B, 1, H, W)
        """
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        # Reshape kernels for conv2d: (out_channels, in_channels, kH, kW)
        k0 = self.kernels['0'].view(1, 1, 1, -1)  # (1, 1, 1, W) - Horizontal
        k1 = self.kernels['1'].view(1, 1, 1, -1)
        k2 = self.kernels['2'].view(1, 1, 1, -1)

        k0_T = k0.transpose(2, 3) # (1, 1, H, 1) - Vertical
        k1_T = k1.transpose(2, 3)
        k2_T = k2.transpose(2, 3)

        pad_size = k0.shape[3] // 2
        
        # Helper for separable convolution with replication padding
        def convolve_sep(input_tensor, k_horz, k_vert):
            # Horizontal pass
            # Pad (Left, Right, Top, Bottom) -> (pad, pad, 0, 0)
            padded_h = F.pad(input_tensor, (pad_size, pad_size, 0, 0), mode='replicate')
            temp = F.conv2d(padded_h, k_horz)
            
            # Vertical pass
            # Pad (0, 0, pad, pad)
            padded_v = F.pad(temp, (0, 0, pad_size, pad_size), mode='replicate')
            out = F.conv2d(padded_v, k_vert)
            return out

        # Ixx: deriv 2 in x, smooth in y -> (I * G''_x) * G_y
        ixx = convolve_sep(image_tensor, k2, k0_T)

        # Iyy: deriv 2 in y, smooth in x -> (I * G_x) * G''_y
        iyy = convolve_sep(image_tensor, k0, k2_T)

        # Ixy: deriv 1 in x, deriv 1 in y -> (I * G'_x) * G'_y
        ixy = convolve_sep(image_tensor, k1, k1_T)
        
        # Also compute Ix and Iy for Steger (1st derivatives)
        # Ix: (I * G'_x) * G_y
        ix = convolve_sep(image_tensor, k1, k0_T)
        
        # Iy: (I * G_x) * G'_y
        iy = convolve_sep(image_tensor, k0, k1_T)

        return ix, iy, ixx, ixy, iyy

    def compute_eigenvalues(self, ixx, ixy, iyy):
        """
        Computes eigenvalues of the Hessian, sorted by absolute magnitude.
        Returns:
            λ1, λ2: Eigenvalues where |λ2| >= |λ1|
        """
        trace = ixx + iyy
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        l_plus = (trace + disc) / 2
        l_minus = (trace - disc) / 2
        
        abs_l_plus = torch.abs(l_plus)
        abs_l_minus = torch.abs(l_minus)
        
        mask_minus_bigger = abs_l_minus > abs_l_plus
        
        λ2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        λ1 = torch.where(mask_minus_bigger, l_plus, l_minus)
        
        return λ1, λ2

    def compute_steger_center(self, ix, iy, ixx, ixy, iyy):
        """
        Implements Steger's line center extraction with rigorous eigenvector sorting.
        Returns:
            t: Subpixel offset along the normal
            valid_mask: Boolean mask of valid points
            nx, ny: Components of the normal vector (eigenvector of max curvature)
            l2: The eigenvalue corresponding to the normal (max curvature)
        """
        # 1. Eigenvalues formulas (λ+, λ-)
        trace = ixx + iyy
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        l_plus = (trace + disc) / 2
        l_minus = (trace - disc) / 2
        
        # 2. Angle and Eigenvectors
        # θ = 0.5 * atan2(2b, a-c)
        # v+ = (cos θ, sin θ)
        # v- = (-sin θ, cos θ)
        theta = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 3. Sort by absolute magnitude to find normal n (max curvature)
        # We want λ2 such that |λ2| >= |λ1|
        abs_l_plus = torch.abs(l_plus)
        abs_l_minus = torch.abs(l_minus)
        
        mask_minus_bigger = abs_l_minus > abs_l_plus
        
        # If |λ-| > |λ+|, then n = v- and λ2 = λ-
        # Else n = v+ and λ2 = λ+
        l2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        
        nx = torch.where(mask_minus_bigger, -sin_t, cos_t)
        ny = torch.where(mask_minus_bigger, cos_t, sin_t)
        
        # 4. Steger computation t
        # t = - (∇r · n) / (n^T H n)
        # By definition, n^T H n = λ2
        dir_deriv_1 = nx * ix + ny * iy
        
        # Avoid division by zero
        mask_nonzero = torch.abs(l2) > 1e-6
        t = torch.zeros_like(l2)
        t[mask_nonzero] = -dir_deriv_1[mask_nonzero] / l2[mask_nonzero]
        
        # Valid mask: |t| <= 0.5
        valid_mask = (torch.abs(t) <= 0.5) & mask_nonzero
        
        return t, valid_mask, nx, ny, l2

if __name__ == "__main__":
    # Simple test
    print("Testing Steger GPU (with Greek notation & Replication padding)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    steger = StegerHessian(σ=2.0, device=device)
    dummy_img = torch.zeros((1, 1, 100, 100), device=device)
    dummy_img[:, :, :, 50] = 1.0
    
    ix, iy, ixx, ixy, iyy = steger.compute_hessian(dummy_img)
    λ1, λ2 = steger.compute_eigenvalues(ixx, ixy, iyy)
    
    # Test new steger center computation
    t, valid, nx, ny, l2_val = steger.compute_steger_center(ix, iy, ixx, ixy, iyy)
    
    print(f"Max Ixx value: {ixx.max().item()}")
    print(f"Max Lambda2 value: {l2_val.max().item()}")
    print(f"Number of valid points: {valid.sum().item()}")
    print("Done.")
