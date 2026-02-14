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
        Computes eigenvalues and eigenvectors of the Hessian.
        Returns:
            λ1, λ2: Eigenvalues
        """
        # Trace = λ1 + λ2 = Ixx + Iyy
        # Det = λ1*λ2 = Ixx*Iyy - Ixy^2
        
        trace = ixx + iyy
        # Discriminant: sqrt(tr^2 - 4*det) = sqrt((Ixx-Iyy)^2 + 4Ixy^2)
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        # Eigenvalues (λ1, λ2)
        l1 = (trace - disc) / 2
        l2 = (trace + disc) / 2
        
        # Sort by absolute value so that |λ1| <= |λ2|
        # This matches the convention used in src/frangi_fusion/hessian.py
        abs_l1 = torch.abs(l1)
        abs_l2 = torch.abs(l2)
        
        swap_mask = abs_l1 > abs_l2
        
        λ1 = torch.where(swap_mask, l2, l1)
        λ2 = torch.where(swap_mask, l1, l2)
        
        return λ1, λ2

    def compute_steger_center(self, ix, iy, ixx, ixy, iyy, λ1, λ2):
        """
        Implements Steger's line center extraction.
        t = - (nx*rx + ny*ry) / (nx^2*rxx + 2*nx*ny*rxy + ny^2*ryy)
        Ideally t = - (n^T * grad) / λ
        """
        
        # Orientation of the structure (perpendicular to normal)
        # Angle θ of the eigenvector corresponding to λ2 (max curvature)
        θ = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        nx = torch.cos(θ)
        ny = torch.sin(θ)
        
        # Term 1: Directional derivative (n^T * grad)
        dir_deriv_1 = nx * ix + ny * iy
        
        # Term 2: Second directional derivative (eigenvalue λ)
        # Ideally this is λ2.
        dir_deriv_2 = λ2 
        
        # Calculate t
        # Avoid division by zero
        mask_nonzero = torch.abs(dir_deriv_2) > 1e-6
        t = torch.zeros_like(dir_deriv_1)
        t[mask_nonzero] = -dir_deriv_1[mask_nonzero] / dir_deriv_2[mask_nonzero]
        
        # Valid mask: |t| <= 0.5 and curvature is sufficiently high
        valid_mask = (torch.abs(t) <= 0.5) & mask_nonzero
        
        return t, valid_mask, nx, ny

if __name__ == "__main__":
    # Simple test
    print("Testing Steger GPU (with Greek notation & Replication padding)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    steger = StegerHessian(σ=2.0, device=device)
    dummy_img = torch.zeros((1, 1, 100, 100), device=device)
    dummy_img[:, :, :, 50] = 1.0
    
    ix, iy, ixx, ixy, iyy = steger.compute_hessian(dummy_img)
    λ1, λ2 = steger.compute_eigenvalues(ixx, ixy, iyy)
    
    print(f"Max Ixx value: {ixx.max().item()}")
    print("Done.")
