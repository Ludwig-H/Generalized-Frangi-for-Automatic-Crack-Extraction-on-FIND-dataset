import torch
import torch.nn.functional as F
import math

class StegerHessian:
    def __init__(self, sigma, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.sigma = sigma
        self.device = device
        self.kernels = self._precompute_kernels()

    def _precompute_kernels(self):
        """
        Creates Gaussian derivative kernels for convolution.
        Returns a dictionary of kernels for 0th, 1st, and 2nd derivatives.
        """
        # Determine kernel size (usually 3*sigma to 6*sigma)
        # We use 4*sigma to be safe and ensure odd size
        size = int(math.ceil(4 * self.sigma)) * 2 + 1
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.device)
        
        # Gaussian function G(x)
        # G(x) = (1 / (sqrt(2pi) * sigma)) * exp(-x^2 / 2sigma^2)
        variance = self.sigma ** 2
        g_x = (1 / (math.sqrt(2 * math.pi) * self.sigma)) * torch.exp(-x**2 / (2 * variance))
        
        # First derivative G'(x) = -x/sigma^2 * G(x)
        g_x_1 = -(x / variance) * g_x
        
        # Second derivative G''(x) = (x^2/sigma^4 - 1/sigma^2) * G(x)
        g_x_2 = ((x**2 / (variance**2)) - (1 / variance)) * g_x

        return {
            '0': g_x,
            '1': g_x_1,
            '2': g_x_2
        }

    def compute_hessian(self, image_tensor):
        """
        Computes Hessian components (Ixx, Ixy, Iyy) for a batch of images.
        image_tensor: (B, 1, H, W)
        """
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        # Reshape kernels for conv2d: (out_channels, in_channels, kH, kW)
        # We need separable convolution for efficiency
        
        k0 = self.kernels['0'].view(1, 1, 1, -1)  # (1, 1, 1, W) - Horizontal
        k1 = self.kernels['1'].view(1, 1, 1, -1)
        k2 = self.kernels['2'].view(1, 1, 1, -1)

        k0_T = k0.transpose(2, 3) # (1, 1, H, 1) - Vertical
        k1_T = k1.transpose(2, 3)
        k2_T = k2.transpose(2, 3)

        # Padding to keep size same
        pad = k0.shape[3] // 2
        
        # Ixx: deriv 2 in x, smooth in y -> (I * G''_x) * G_y
        # Step 1: Convolve with G''_x
        ixx_temp = F.conv2d(image_tensor, k2, padding=(0, pad))
        # Step 2: Convolve result with G_y
        ixx = F.conv2d(ixx_temp, k0_T, padding=(pad, 0))

        # Iyy: deriv 2 in y, smooth in x -> (I * G_x) * G''_y
        iyy_temp = F.conv2d(image_tensor, k0, padding=(0, pad))
        iyy = F.conv2d(iyy_temp, k2_T, padding=(pad, 0))

        # Ixy: deriv 1 in x, deriv 1 in y -> (I * G'_x) * G'_y
        ixy_temp = F.conv2d(image_tensor, k1, padding=(0, pad))
        ixy = F.conv2d(ixy_temp, k1_T, padding=(pad, 0))
        
        # Also compute Ix and Iy for Steger (1st derivatives)
        # Ix: (I * G'_x) * G_y
        ix = F.conv2d(ixy_temp, k0_T, padding=(pad, 0))
        
        # Iy: (I * G_x) * G'_y
        iy = F.conv2d(iyy_temp, k1_T, padding=(pad, 0))

        return ix, iy, ixx, ixy, iyy

    def compute_eigenvalues(self, ixx, ixy, iyy):
        """
        Computes eigenvalues and eigenvectors of the Hessian.
        Returns:
            lambda1, lambda2: Eigenvalues (lambda1 < lambda2 usually for sorting by magnitude, 
                              but here we care about curvature. 
                              For dark cracks on bright background, we look for positive max curvature)
            v1, v2: Eigenvectors
        """
        # Construct the Hessian matrix for each pixel
        # H = [[ixx, ixy], [ixy, iyy]]
        # Stack to shape (B, H, W, 2, 2)
        
        # Efficient analytical solution for 2x2 symmetric matrix
        # Trace = l1 + l2 = Ixx + Iyy
        # Det = l1*l2 = Ixx*Iyy - Ixy^2
        
        trace = ixx + iyy
        det = ixx * iyy - ixy**2
        
        # Discriminant: sqrt(tr^2 - 4*det) = sqrt((Ixx+Iyy)^2 - 4(IxxIyy - Ixy^2))
        # = sqrt((Ixx-Iyy)^2 + 4Ixy^2)
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        # Eigenvalues
        l1 = (trace - disc) / 2
        l2 = (trace + disc) / 2
        
        # We need to sort by absolute magnitude for "vesselness", 
        # but for Steger we need the direction of MAX curvature (perpendicular to line).
        # For dark lines (valleys), the curvature across the line is POSITIVE and Large.
        # Ideally l2 is the large positive one.
        
        # Eigenvectors
        # If Ixy is not zero: v = [lambda - Iyy, Ixy] or [Ixy, lambda - Ixx]
        # We can use atan2 for orientation.
        # Angle of the eigenvector corresponding to l2 (max curvature):
        # theta = 0.5 * atan2(2*Ixy, Ixx - Iyy)
        
        return l1, l2

    def compute_steger_center(self, ix, iy, ixx, ixy, iyy, l1, l2):
        """
        Implements Steger's line center extraction.
        Equation: (t * n_x + n_y) . gradient = 0 ?? No.
        
        Steger 1998:
        The Taylor expansion of the image profile in direction n (normal to line) is:
        f(t) = r + t*n^T*grad + 0.5*t^2*n^T*H*n
        We want f'(t) = 0 => n^T*grad + t*n^T*H*n = 0
        t = - (n^T * grad) / (n^T * H * n)
        
        Where n is the eigenvector corresponding to the maximum absolute eigenvalue (curvature).
        Let n = (nx, ny).
        t = - (nx*rx + ny*ry) / (nx^2*rxx + 2*nx*ny*rxy + ny^2*ryy)
        
        Wait, n is eigenvector of H, so H*n = lambda*n.
        So n^T * H * n = n^T * (lambda * n) = lambda * (n^T * n) = lambda.
        
        So t = - (n^T * grad) / lambda.
        
        Check validity: t \in [-0.5, 0.5]
        """
        
        # 1. Identify dominant eigenvector n corresponding to max absolute curvature.
        # We assume dark lines => positive curvature.
        # If l2 is the largest eigenvalue:
        # Vector v2 corresponds to l2.
        
        # We can compute v2 explicitly.
        # v2_x = l2 - iyy
        # v2_y = ixy
        # Norm = sqrt(v2_x^2 + v2_y^2)
        
        # Handling the case where ixy is small (diagonal)
        # ...
        
        # Let's use analytical orientation calculation to be safe.
        # Orientation of the structure (perpendicular to normal)
        # angle_normal = 0.5 * atan2(2*ixy, ixx - iyy)
        # nx = cos(angle_normal), ny = sin(angle_normal)
        
        # However, we must ensure l2 is the one corresponding to this angle.
        # The angle from atan2 corresponds to the eigenvector with LARGER eigenvalue if Ixx > Iyy?
        # Let's check: 
        # if Ixx=10, Iyy=0, Ixy=0 -> atan2(0, 10) = 0. nx=1, ny=0. Correct.
        # if Ixx=0, Iyy=10, Ixy=0 -> atan2(0, -10) = pi. 0.5*pi = pi/2. nx=0, ny=1. Correct.
        
        # So yes, angle corresponds to max eigenvalue direction (l2).
        
        angle = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        nx = torch.cos(angle)
        ny = torch.sin(angle)
        
        # Term 1: Directional derivative (n^T * grad)
        # = nx * Ix + ny * Iy
        dir_deriv_1 = nx * ix + ny * iy
        
        # Term 2: Second directional derivative (eigenvalue lambda)
        # Ideally this is l2.
        # But let's compute it to be consistent with nx, ny sign.
        # dir_deriv_2 = nx*nx*ixx + 2*nx*ny*ixy + ny*ny*iyy
        # (Should match l2)
        dir_deriv_2 = l2 
        
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
    print("Testing Steger GPU...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    steger = StegerHessian(sigma=2.0, device=device)
    dummy_img = torch.zeros((1, 1, 100, 100), device=device)
    # Create a vertical line
    dummy_img[:, :, :, 50] = 1.0
    
    ix, iy, ixx, ixy, iyy = steger.compute_hessian(dummy_img)
    l1, l2 = steger.compute_eigenvalues(ixx, ixy, iyy)
    
    # For a bright line, curvature is negative.
    # We might need to handle sign. 
    # Usually Frangi filters invert image or look for negative eigenvalues.
    
    print(f"Max Ixx value: {ixx.max().item()}")
    print("Done.")
