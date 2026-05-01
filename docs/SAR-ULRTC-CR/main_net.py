import torch
from torch import nn
import cv2
import torch.nn.functional as F
import torchvision
import numpy as np
# import torchmaxflow
from torch.linalg import lstsq
import basicblock as B

# Helper function for padding
def pad_tensor(tensor, radius):
    """Pads a tensor using reflection padding."""
    # Ensure tensor is at least 4D for F.pad
    needs_unsqueeze = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        needs_unsqueeze = True
    
    padded = F.pad(tensor, (radius, radius, radius, radius), mode='reflect')
    
    if needs_unsqueeze:
        padded = padded.squeeze(0)
    return padded


# Helper function for box filtering (window summation)
def box_filter(tensor, radius):
    """Performs box filtering (summation over a window) using convolution.
    Args:
        tensor (torch.Tensor): Input tensor (B, C, H, W).
        radius (int): Radius of the window (window size is 2*radius + 1).
    Returns:
        torch.Tensor: Filtered tensor (B, C, H, W).
    """
    if tensor.ndim != 4:
        raise ValueError(f"box_filter expects a 4D tensor (B, C, H, W), got {tensor.ndim}D")
        
    B, C, H, W = tensor.shape
    kernel_size = 2 * radius + 1
    # Use depthwise convolution for independent channel filtering
    # Ensure kernel has same dtype and device
    kernel = torch.ones((C, 1, kernel_size, kernel_size), dtype=tensor.dtype, device=tensor.device)
    # Padding handled by conv2d
    # Input to conv2d should be 4D
    return F.conv2d(tensor, kernel, padding=radius, groups=C)

class FastMattingSolver:
    def __init__(self, image, trimap, radius=5, epsilon=1e-5, lambda_val=100.0, device='cuda'):
        """
        Args:
            image (torch.Tensor): Input RGB image (3, H, W), range [0, 1].
            trimap (torch.Tensor): Input trimap (1, H, W), values: 0 (BG), 0.5 (Unknown), 1 (FG).
            radius (int): Window radius r for the Matting Laplacian.
            epsilon (float): Regularization term for matrix inversion (Eq. 5).
            lambda_val (float): Weight for the data term (Eq. 6).
            device (str): 'cuda' or 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

        # Ensure image and trimap are on the correct device from the start
        self.image = image.to(self.device) # Should be [3, H, W]
        self.trimap = trimap.to(self.device) # Should be [1, H, W]

        if self.image.ndim != 3 or self.image.shape[0] != 3:
             raise ValueError(f"Input image must be 3D (3, H, W), got {self.image.shape}")
        if self.trimap.ndim != 3 or self.trimap.shape[0] != 1:
             raise ValueError(f"Input trimap must be 3D (1, H, W), got {self.trimap.shape}")
        if self.image.shape[1:] != self.trimap.shape[1:]:
             raise ValueError(f"Image ({self.image.shape[1:]}) and trimap ({self.trimap.shape[1:]}) spatial dimensions must match")


        self.radius = radius
        self.epsilon = epsilon
        self.lambda_val = lambda_val

        self.H, self.W = self.image.shape[1:]
        self.window_pixels = (2 * self.radius + 1) ** 2 # Renamed from window_size

        # Identify unknown pixels
        self.unknown_mask = (self.trimap > 0.01) & (self.trimap < 0.99) # Shape [1, H, W]
        self.known_mask = ~self.unknown_mask # Shape [1, H, W]

        # Prepare constraint vector beta and diagonal matrix D values
        self.beta = self.trimap.clone().squeeze(0) # H, W
        # Set unknowns in beta to 0, it doesn't matter for the RHS λDβ
        self.beta[self.unknown_mask.squeeze(0)] = 0

        # D_val corresponds to the diagonal entries of λD
        self.D_val = self.known_mask.float() * self.lambda_val # Shape [1, H, W]

        # Get indices and count of unknown pixels
        self.unknown_indices_flat = torch.where(self.unknown_mask.view(-1))[0]
        self.N_unknown = len(self.unknown_indices_flat)
        # if self.N_unknown == 0:
        #      print("Warning: No unknown pixels found in the trimap.")
        # print(f"Image size: {self.H}x{self.W}, Unknown pixels: {self.N_unknown}")

        # Map flat unknown indices to 2D coordinates (optional, mainly for debugging)
        # self.unknown_coords_y = self.unknown_indices_flat // self.W
        # self.unknown_coords_x = self.unknown_indices_flat % self.W

        # Precompute padded image for window operations
        # Add batch dim for padding and box_filter
        self.image_batch = self.image.unsqueeze(0) # [1, 3, H, W]
        self.image_padded = pad_tensor(self.image_batch, self.radius) # [1, 3, H+2r, W+2r]

        # Precompute terms needed for Lp calculation (Eq. 9)
        # Mean I (μk) and I*I^T
        # print("Precomputing window means and covariances...")

        # --- FIX 1: Crop after box_filter ---
        sum_I_padded = box_filter(self.image_padded, self.radius) # [1, 3, H+2r, W+2r]
        self.sum_I = sum_I_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 3, H, W]
        self.mean_I = self.sum_I / self.window_pixels # μk [1, 3, H, W]

        # Compute I*I^T per pixel, then sum over windows
        I_unpadded = self.image_batch # [1, 3, H, W]
        # Efficient outer product calculation
        I_permuted = I_unpadded.permute(0, 2, 3, 1) # [1, H, W, 3]
        outer_I = torch.matmul(I_permuted.unsqueeze(4), I_permuted.unsqueeze(3)) # [1, H, W, 3, 3]
        outer_I_flat = outer_I.view(1, self.H, self.W, 9).permute(0, 3, 1, 2) # [1, 9, H, W]

        outer_I_padded = pad_tensor(outer_I_flat, self.radius) # [1, 9, H+2r, W+2r]

        # --- FIX 2: Crop after box_filter ---
        sum_outer_I_padded = box_filter(outer_I_padded, self.radius) # [1, 9, H+2r, W+2r]
        self.sum_outer_I = sum_outer_I_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 9, H, W]
        self.mean_outer_I = self.sum_outer_I / self.window_pixels # E[I*I^T]k [1, 9, H, W]

        # Compute Covariance Matrix Σk (Eq. 5)
        # Σk = E[I*I^T]k - μk * μk^T
        mean_I_permuted = self.mean_I.permute(0, 2, 3, 1) # [1, H, W, 3]
        outer_mean_I = torch.matmul(mean_I_permuted.unsqueeze(4), mean_I_permuted.unsqueeze(3)) # [1, H, W, 3, 3]
        outer_mean_I_flat = outer_mean_I.view(1, self.H, self.W, 9).permute(0, 3, 1, 2) # [1, 9, H, W] (μk * μk^T)

        cov_k_flat = self.mean_outer_I - outer_mean_I_flat # [1, 9, H, W]
        # Reshape to [1, H, W, 3, 3] for easier matrix operations later
        self.cov_k = cov_k_flat.view(1, 3, 3, self.H, self.W).permute(0, 3, 4, 1, 2) # [1, H, W, 3, 3]

        # Compute Delta_k = Σk + (eps/|ωk|) * U (Eq. 9)
        identity = torch.eye(3, 3, device=self.device).view(1, 1, 1, 3, 3)
        self.delta_k = self.cov_k + (self.epsilon / self.window_pixels) * identity # [1, H, W, 3, 3]
        # print("Precomputation finished.")


    def _compute_laplacian_step1(self, p_img):
        """ Computes a*k and b*k (Eq. 9, 10) for a given vector p.
        Args:
            p_img (torch.Tensor): Vector p represented as an image (1, 1, H, W).
        Returns:
            a_star (torch.Tensor): Affine coefficient a* (1, H, W, 3, 1).
            b_star (torch.Tensor): Affine coefficient b* (1, 1, H, W).
        """
        if p_img.ndim != 4 or p_img.shape[:2] != (1,1):
             raise ValueError(f"p_img must be 4D (1, 1, H, W), got {p_img.shape}")

        p_padded = pad_tensor(p_img, self.radius) # [1, 1, H+2r, W+2r]

        # Compute window sums needed for Eq. 9
        # --- FIX 3: Crop after box_filter ---
        sum_p_padded = box_filter(p_padded, self.radius) # [1, 1, H+2r, W+2r]
        sum_p = sum_p_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 1, H, W]
        mean_p = sum_p / self.window_pixels # pk_mean [1, 1, H, W]

        # Compute I*p per pixel, then sum over windows
        # self.image_batch is [1, 3, H, W], p_img is [1, 1, H, W]
        Ip = self.image_batch * p_img # [1, 3, H, W]
        Ip_padded = pad_tensor(Ip, self.radius) # [1, 3, H+2r, W+2r]

        # --- FIX 4: Crop after box_filter ---
        sum_Ip_padded = box_filter(Ip_padded, self.radius) # [1, 3, H+2r, W+2r]
        sum_Ip = sum_Ip_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 3, H, W]
        mean_Ip = sum_Ip / self.window_pixels # E[I*p]k [1, 3, H, W]

        # Calculate RHS of Eq. 9: (E[I*p]k - μk * pk_mean)
        # self.mean_I: [1, 3, H, W]
        # mean_p: [1, 1, H, W]
        rhs = mean_Ip - self.mean_I * mean_p # [1, 3, H, W]
        rhs = rhs.permute(0, 2, 3, 1).unsqueeze(-1) # [1, H, W, 3, 1]

        # Solve for a*: Δk * a* = rhs => a* = Δk^-1 * rhs (Eq. 9)
        # self.delta_k: [1, H, W, 3, 3]
        # Using torch.linalg.lstsq which handles batches and is robust
        delta_flat = self.delta_k.view(-1, 3, 3) # [HW, 3, 3]
        rhs_flat = rhs.view(-1, 3, 1)           # [HW, 3, 1]

        # Use lstsq for potentially better stability than solve with epsilon
        try:
             solution = lstsq(delta_flat, rhs_flat)
             a_star_flat = solution.solution
        except torch.linalg.LinAlgError as e:
             print(f"Warning: Linear system solve failed: {e}. Check epsilon value or image content.")
             # Fallback or error handling needed, e.g., return zeros or raise
             a_star_flat = torch.zeros_like(rhs_flat)


        a_star = a_star_flat.view(1, self.H, self.W, 3, 1) # [1, H, W, 3, 1]

        # Calculate b* (Eq. 10): b* = pk_mean - a*^T * μk
        # a_star: [1, H, W, 3, 1] -> permute to [1, H, W, 1, 3]
        # self.mean_I: [1, 3, H, W] -> permute to [1, H, W, 3, 1]
        a_star_T = a_star.permute(0, 1, 2, 4, 3) # [1, H, W, 1, 3]
        mean_I_reshaped = self.mean_I.permute(0, 2, 3, 1).unsqueeze(-1) # [1, H, W, 3, 1]
        aT_mu = torch.matmul(a_star_T, mean_I_reshaped).squeeze(-1).squeeze(-1) # [1, H, W]

        b_star = mean_p.squeeze(1) - aT_mu # [1, H, W]
        b_star = b_star.unsqueeze(1) # Restore channel dim -> [1, 1, H, W]

        return a_star, b_star

    def _compute_laplacian_step2(self, a_star, b_star, p_img):
        """ Computes Lp using Eq. 11 from precomputed a* and b*.
        Args:
            a_star (torch.Tensor): Affine coefficient a* (1, H, W, 3, 1).
            b_star (torch.Tensor): Affine coefficient b* (1, 1, H, W).
            p_img (torch.Tensor): Vector p represented as an image (1, 1, H, W).
        Returns:
            Lp (torch.Tensor): Result of L * p as an image (1, 1, H, W).
        """
         # Calculate sum_k_in_wi(a*k) and sum_k_in_wi(b*k)
        # This is equivalent to box filtering a* and b* over windows centered at i
        # Input a* represents coefficients *for each window k*
        # We need sum over k where i is *in* window k (k ∈ ωi)

        # Reshape a* for filtering: [1, H, W, 3, 1] -> [1, 3, H, W]
        a_star_img = a_star.squeeze(-1).permute(0, 3, 1, 2) # [1, 3, H, W]
        a_star_img_padded = pad_tensor(a_star_img, self.radius) # [1, 3, H+2r, W+2r]

        # --- FIX 5: Crop after box_filter ---
        sum_a_star_padded = box_filter(a_star_img_padded, self.radius) # [1, 3, H+2r, W+2r]
        sum_a_star = sum_a_star_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 3, H, W]

        # b_star is [1, 1, H, W]
        b_star_padded = pad_tensor(b_star, self.radius) # [1, 1, H+2r, W+2r]

        # --- FIX 6: Crop after box_filter ---
        sum_b_star_padded = box_filter(b_star_padded, self.radius) # [1, 1, H+2r, W+2r]
        sum_b_star = sum_b_star_padded[:, :, self.radius:-self.radius, self.radius:-self.radius] # [1, 1, H, W]

        # Calculate Lp using Eq. 11:
        # (Lp)i = |ωi| * pi - ( sum_{k in ωi}(a*k^T * Ii) + sum_{k in ωi}(b*k) )
        #       = |ωi| * pi - ( (sum_{k in ωi}a*k)^T * Ii + sum_{k in ωi}b*k )
        # Note: |ωi| is self.window_pixels

        # sum_a_star: [1, 3, H, W] -> permute to [1, H, W, 1, 3]
        # self.image_batch (Ii): [1, 3, H, W] -> permute to [1, H, W, 3, 1]
        sum_a_star_T = sum_a_star.permute(0, 2, 3, 1).unsqueeze(-2) # [1, H, W, 1, 3]
        I_reshaped = self.image_batch.permute(0, 2, 3, 1).unsqueeze(-1) # [1, H, W, 3, 1]

        term1 = torch.matmul(sum_a_star_T, I_reshaped).squeeze(-1) # [1, H, W, 1]
        term1 = term1.permute(0, 3, 1, 2) # [1, 1, H, W]

        # sum_b_star: [1, 1, H, W]

        # Lp = |ω|*p - (term1 + term2)
        Lp = (self.window_pixels * p_img) - (term1 + sum_b_star) # [1, 1, H, W]
        return Lp

    def matvec(self, p_flat):
        """ Computes the matrix-vector product (L + λD) * p for CG.
        Args:
            p_flat (torch.Tensor): Flattened vector p corresponding to unknown pixels (N_unknown,).
        Returns:
            Ap_flat (torch.Tensor): Flattened result vector (N_unknown,).
        """
        # Create a full image representation of p, setting known pixels to 0
        p_img = torch.zeros((1, 1, self.H, self.W), dtype=p_flat.dtype, device=self.device)
        if self.N_unknown > 0: # Avoid index error if no unknowns
             p_img.view(-1)[self.unknown_indices_flat] = p_flat

        # Compute Lp using the efficient two-step approach
        a_star, b_star = self._compute_laplacian_step1(p_img)
        Lp = self._compute_laplacian_step2(a_star, b_star, p_img) # [1, 1, H, W]

        # Compute λDp
        # self.D_val is [1, H, W], needs unsqueeze. p_img is [1, 1, H, W]
        lambda_Dp = self.D_val.unsqueeze(1) * p_img # [1, 1, H, W]

        # Compute (L + λD)p
        Ap = Lp + lambda_Dp # [1, 1, H, W]

        # Extract the values corresponding to unknown pixels
        if self.N_unknown > 0:
             Ap_flat = Ap.view(-1)[self.unknown_indices_flat]
        else:
             Ap_flat = torch.empty(0, dtype=p_flat.dtype, device=self.device) # Return empty tensor
        return Ap_flat

    def compute_Lp(self, p_img):
        """ Computes only the Lp part of the matrix-vector product.
            Corresponds to applying the Matting Laplacian L to p_img.
        Args:
            p_img (torch.Tensor): Vector p represented as an image (1, 1, H, W).
                                  Assumed to be on the correct device.
        Returns:
            Lp (torch.Tensor): Result of L * p as an image (1, 1, H, W).
        """
        # Ensure p_img has correct shape and device
        if p_img.ndim != 4 or p_img.shape[:2] != (1,1):
             # Attempt to reshape if needed, assuming H, W are correct
             try:
                 p_img = p_img.view(1, 1, self.H, self.W)
             except RuntimeError as e:
                  raise ValueError(f"p_img cannot be reshaped to (1, 1, {self.H}, {self.W}). Got shape {p_img.shape}. Error: {e}")

        # Compute Lp using the efficient two-step approach
        try:
            a_star, b_star = self._compute_laplacian_step1(p_img)
            Lp = self._compute_laplacian_step2(a_star, b_star, p_img) # [1, 1, H, W]
        except Exception as e:
             print(f"Error during Lp calculation steps: {e}")
             import traceback
             traceback.print_exc()
             # Return zero tensor as fallback? Or re-raise?
             # Returning zeros might stall CG silently. Re-raising is safer.
             raise e
        return Lp

    def solve(self, tol=1e-5, max_iter=200):
        """ Solves for the alpha matte using Conjugate Gradient. """
        # print("Starting Conjugate Gradient solver...")
        if self.N_unknown == 0:
            # print("No unknown pixels to solve for. Returning trimap.")
            # Return trimap values directly (ensure knowns are 0 or 1)
            alpha_final = self.trimap.clone().squeeze(0) # H, W
            alpha_final[alpha_final < 0.5] = 0.0
            alpha_final[alpha_final >= 0.5] = 1.0
            return alpha_final.unsqueeze(0) # 1, H, W

        # --- Calculate Correct Right-Hand Side b = -L_uk α_k ---
        # The system we solve is L_uu α_u = -L_uk α_k
        # print("Calculating RHS for CG...")
        # start_rhs_time = time.time()
        # Create image with known alpha values (alpha_k), unknowns are 0
        alpha_known_img = torch.zeros((1, 1, self.H, self.W), dtype=self.image.dtype, device=self.device)
        known_mask_bool = self.known_mask.squeeze(0) # H, W boolean mask
        # self.beta (H, W) already has known values (0 or 1) and 0 for unknowns after init
        alpha_known_img[0, 0, known_mask_bool] = self.beta[known_mask_bool]

        # Compute L * alpha_known_img (this computes L_uk * α_k + L_uu * 0)
        L_alpha_known = self.compute_Lp(alpha_known_img) # Use the helper

        # Extract unknown components and negate: b = -(L * alpha_known)_u
        rhs_flat = -L_alpha_known.view(-1)[self.unknown_indices_flat] # [N_unknown]
        # end_rhs_time = time.time()
        # print(f"RHS calculated in {end_rhs_time - start_rhs_time:.2f} seconds.")

        rhs_norm_val = torch.norm(rhs_flat).item()
        # print(f"RHS norm: {rhs_norm_val:.4e}")
        # if rhs_norm_val < 1e-9:
            #  print("Warning: Calculated RHS is close to zero. Result might not change much.")
            #  print("This could happen if known regions have minimal influence on unknowns,")
            #  print("or if epsilon/radius/lambda parameters are suboptimal.")
             # We can still proceed, CG will try to solve Ax=0

        # --- Initial guess for unknowns ---
        alpha_init = self.trimap.clone() # [1, H, W]
        # Initialize unknowns to 0.5 or another value (e.g., propagate nearest known)
        # Using 0.5 is simple and common.
        alpha_init[self.unknown_mask] = 0.5
        # x represents the current estimate for alpha_u (unknowns only)
        x = alpha_init.view(-1)[self.unknown_indices_flat] # [N_unknown]

        # --- Conjugate Gradient Setup ---
        # start_cg_time = time.time()
        # Initial residual r = b - Ax
        # Note: Ax = matvec(x) calculates L_uu * x_u because lambda_Dp is zero for unknowns
        try:
            Ax = self.matvec(x)
        except Exception as e:
             print(f"Error during initial matvec calculation: {e}")
             raise e # Propagate error

        r = rhs_flat - Ax
        p = r.clone()
        try:
             rsold = torch.dot(r, r)
        except RuntimeError as e:
             print(f"Error calculating initial dot product (rsold): {e}")
             print(f"Residual tensor 'r': dtype={r.dtype}, shape={r.shape}, has_nan={torch.isnan(r).any()}, has_inf={torch.isinf(r).any()}")
             raise e

        # Calculate norm of the *actual* RHS for relative tolerance check
        norm_rhs = torch.sqrt(torch.dot(rhs_flat, rhs_flat))
        if norm_rhs == 0:
            # print("RHS is zero. Solution is likely zero.")
            norm_rhs = 1.0 # Avoid division by zero in relative residual calculation

        initial_residual_norm = torch.sqrt(rsold)
        # print(f"Initial Residual Norm: {initial_residual_norm.item():.4e}, Relative: {initial_residual_norm/norm_rhs:.4e}")

        if initial_residual_norm < tol:
            #  print("Initial guess is already within tolerance.")
             max_iter = 0 # Skip CG loop

        iter_num = 0
        # --- Conjugate Gradient Loop ---
        for i in range(max_iter):
            iter_num = i + 1
            try:
                # Ap = L_uu * p_u
                Ap = self.matvec(p)
            except Exception as e:
                # print(f"Error during matvec calculation in CG loop (iter {i+1}): {e}")
                # Optionally break or try to recover, but re-raising is safer
                raise e

            pAp = torch.dot(p, Ap)

            # Safeguard for potential division by zero or negative pAp (matrix might not be perfectly SPD due to numerics)
            if abs(pAp.item()) < 1e-12:
                #  print(f"\nWarning: pAp close to zero ({pAp.item():.2e}) at iter {i+1}. Stopping CG to avoid division by zero.")
                #  print("This might indicate issues with matrix conditioning or convergence.")
                 break
            if pAp.item() < 0:
                #  print(f"\nWarning: pAp is negative ({pAp.item():.2e}) at iter {i+1}. Matrix might not be positive definite. Stopping CG.")
                 break

            alpha_cg = rsold / pAp
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap
            rsnew = torch.dot(r, r)
            residual_norm = torch.sqrt(rsnew)

            # if (i + 1) % 10 == 0 or i == 0: # Print progress every 10 iterations
            #      print(f"Iter: {i+1:03d}, Residual Norm: {residual_norm.item():.4e}, Relative: {residual_norm/norm_rhs:.4e}")

            if residual_norm / norm_rhs < tol:
                # print(f"\nConverged in {iter_num} iterations (Relative residual {residual_norm/norm_rhs:.2e} < tolerance {tol:.2e})")
                break

            # Update direction vector p
            beta_cg = rsnew / rsold
            p = r + beta_cg * p
            rsold = rsnew

            # Check for NaN/Inf in residual (indicates divergence)
            if torch.isnan(rsnew) or torch.isinf(rsnew):
                #  print(f"\nWarning: CG diverged (NaN/Inf residual) at iter {i+1}. Check parameters (lambda, epsilon, radius).")
                 # You might want to return the last valid 'x' or raise an error
                 break
        else:
            # Loop finished without break (i.e., max_iter reached)
            # print(f"\nWarning: CG did not converge within {max_iter} iterations.")
            # print(f"Final Residual Norm: {residual_norm.item():.4e}, Relative: {residual_norm/norm_rhs:.4e} (Tolerance: {tol:.2e})")
            pass


        # end_cg_time = time.time()
        # print(f"CG loop finished in {end_cg_time - start_cg_time:.2f} seconds.")

        # --- Combine result ---
        alpha_final = self.trimap.clone().squeeze(0) # H, W, contains 0, 0.5, 1
        # Clamp CG results to [0, 1] for stability before assigning
        x_clamped = torch.clamp(x, 0.0, 1.0)
        # Place solved unknown values into the alpha map
        alpha_final.view(-1)[self.unknown_indices_flat] = x_clamped

        # Optional: Ensure known regions remain *exactly* 0 or 1 *after* solving
        # This might slightly contradict the solved values near the boundary, but enforces hard constraints.
        # Use the original trimap thresholds for consistency.
        alpha_final[self.trimap.squeeze(0) < 0.1] = 0.0
        alpha_final[self.trimap.squeeze(0) > 0.9] = 1.0
        # It might be better *not* to do this if the solved values provide smoother transitions. Test visually.

        return alpha_final.unsqueeze(0) # Return as [1, H, W]


class SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(x))))
        return att_map

class SpatialAttentionModule_SAR(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule_SAR, self).__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats-2-1, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        att_map = self.att2(self.relu(self.att1(x)))
        return att_map

class SAM(nn.Module): # Spatial Attention Module
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        # output = F.sigmoid(output) * x 
        return F.sigmoid(output) 

class PALayer(nn.Module): # Pixel Attention Layer
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return y

class simam_module(torch.nn.Module): # Simple Attention Module
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs
    
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB_smilecr(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_smilecr, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=13, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB_2(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_2, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=15, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB_3(nn.Module): #with cloud threshold
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_3, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=13, kernel_size=3, stride=1, padding=1)
        self.cloud_threshold = nn.Threshold(0.2, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        x = self.cloud_threshold(x)
        return x

class RDB_mask(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB_mask_2(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask_2, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to output
        return x

class RDB_mask_2_smilecr(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask_2_smilecr, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to output
        return x

class ProximalBlock(nn.Module):
    def __init__(self, out_channels):
        super(ProximalBlock, self).__init__()
        self.out_channels = out_channels
        self.proximal = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output = self.proximal(x)
        if self.out_channels == 1:
            output = self.sigmoid(output)
        return output

class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x

class RPCA_Block(nn.Module):
    def __init__(self):
        super(RPCA_Block, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, C_k):
        C = torch.fft.fft(torch.squeeze(C_k), n=C_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(C, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), C).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)


    def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):

        # Update C
        psi_c = self.mu + self.alpha
        Psi_C = (L1 - L2 + self.mu * Omega - self.mu * E - self.mu * T + self.alpha * P)
        C_k = torch.div(torch.mul(W, self.tensor_product(L, R)) + Psi_C, W + psi_c)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, C_k)

        # Update E
        psi_e = self.mu + self.beta
        Psi_E = (L1 - L3 + self.mu * Omega - self.mu * C_k - self.mu * T + self.beta * Q) / psi_e
        E_k = torch.mul(torch.sign(Psi_E), nn.functional.relu(torch.abs(Psi_E) - self.lamb / psi_e))

        # Update T
        Y = Omega - C_k - E_k + L1 / self.mu
        T_k = torch.mul(Y, Omega_C) + \
              torch.mul(Y, Omega) * torch.min(torch.tensor(1.).cuda(), \
                        self.delta / (torch.norm(torch.mul(Y, Omega), 'fro') + 1e-6))

        # Update P
        P_k = self.Proximal_P(C_k + L2 / (self.alpha + 1e-6))

        # Update Q
        Q_k = self.Proximal_Q(E_k + L3 / (self.beta + 1e-6))

        # Update Lambda
        L1_k = L1 + self.mu * (Omega - C_k - E_k - T_k)
        L2_k = L2 + self.alpha * (C_k - P_k)
        L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, C_k, E_k, T_k, P_k, Q_k, L1_k, L2_k, L3_k

class RPCA_Block_2(nn.Module):
    def __init__(self):
        super(RPCA_Block_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_2(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, C_k):
        C = torch.fft.fft(torch.squeeze(C_k), n=C_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(C, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), C).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)


    def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):

        # Update C
        psi_c = self.mu + self.alpha
        Psi_C = (L1 - L2 + self.mu * Omega - self.mu * E - self.mu * T + self.alpha * P)
        C_k = torch.div(torch.mul(W, self.tensor_product(L, R)) + Psi_C, W + psi_c)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, C_k)

        # Update E
        psi_e = self.mu + self.beta
        Psi_E = (L1 - L3 + self.mu * Omega - self.mu * C_k - self.mu * T + self.beta * Q) / psi_e
        E_k = torch.mul(torch.sign(Psi_E), nn.functional.relu(torch.abs(Psi_E) - self.lamb / psi_e))

        # Update T
        Y = Omega - C_k - E_k + L1 / self.mu
        T_k = torch.mul(Y, Omega_C) + \
              torch.mul(Y, Omega) * torch.min(torch.tensor(1.).cuda(), \
                        self.delta / (torch.norm(torch.mul(Y, Omega), 'fro') + 1e-6))

        # Update P
        P_k = self.Proximal_P(C_k + L2 / (self.alpha + 1e-6))

        # Update Q
        Q_k = self.Proximal_Q(E_k + L3 / (self.beta + 1e-6))

        # Update Lambda
        L1_k = L1 + self.mu * (Omega - C_k - E_k - T_k)
        L2_k = L2 + self.alpha * (C_k - P_k)
        L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, C_k, E_k, T_k, P_k, Q_k, L1_k, L2_k, L3_k

class RPCA_Block_new(nn.Module):
    def __init__(self):
        super(RPCA_Block_new, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        # self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L1, L2, Omega, W):

        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + self.alpha * G - L1, M ** 2 + W + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + self.beta * H - L2, X ** 2 + self.beta)
        # Threshold M_k
        M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L1_k, L2_k

class RPCA_Block_new_2(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_ver2(nn.Module): # with Unet
    def __init__(self):
        super(RPCA_Block_new_2_ver2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_ver2_1(nn.Module): # with Unet multi batchsize
    def __init__(self):
        super(RPCA_Block_new_2_ver2_1, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    # def tensor_product(self, L, R):
    #     Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
    #     Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
    #     Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
    #     return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    # def decom_solution(self, L_k, R_k, X_k):
    #     X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
    #     L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
    #     R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

    #     Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
    #                       torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

    #     Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
    #                       torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

    #     return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
    #            torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # print('M_k:',M_k.shape)
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim=True) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

class RPCA_Block_new_3_ver2(nn.Module): # with U-net
    def __init__(self):
        super(RPCA_Block_new_3_ver2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet(in_channels=13, out_channels=13)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class UNet_full(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_full, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.encoder4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.decoder4 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))
        middle = self.middle(self.pool3(enc3))

        # dec4 = self.up4(middle)
        # dec4 = torch.cat((dec4, enc3), dim=1)
        # dec4 = self.decoder4(dec4)

        dec3 = self.up3(middle)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final(dec1)
    
class RPCA_Block_new_3_ver3(nn.Module): # with U-net full
    def __init__(self):
        super(RPCA_Block_new_3_ver3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_ver3_2(nn.Module): # with U-net full
    def __init__(self):
        super(RPCA_Block_new_3_ver3_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.2, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_ver3_3(nn.Module): # with mask
    def __init__(self):
        super(RPCA_Block_new_3_ver3_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(C_k, dim=1) > 0.2, torch.ones_like(C_k), torch.zeros_like(C_k))
        M_k = M
        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_ver3_SAR(nn.Module): # with U-net full
    def __init__(self):
        super(RPCA_Block_new_3_ver3_SAR, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet_full(in_channels=15, out_channels=15)
        self.Proximal_H = RDB_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W):
        
        # Y is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.div(torch.mul(Y,X) - torch.mul(C, X) - self.lamb + \
        #                 self.mu * torch.mul(Y - X,Y - X) + torch.mul(L0,(Y-X)) + self.beta * H - L2,\
        #                 torch.mul(X,X)+ self.mu * torch.mul(Y - X, Y - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(X_k + L1 / (self.alpha + 1e-6))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_ver3_SAR_CS(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_3_ver3_SAR_CS, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp):
        
        # Y is Y | W is F_att(Y)
        # Update X
        
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M, cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)
        
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - torch.mul(self.ones,self.lamb) + torch.mul(self.beta, H) - L2, self.ones + torch.mul(self.ones, self.beta))

        # Update M
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_ver3_SAR_CS_2(nn.Module): # with U-net full and SAR concat X 
    def __init__(self):
        super(RPCA_Block_new_3_ver3_SAR_CS_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_G = UNet_full(in_channels=15, out_channels=13)
        self.Proximal_H = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Y, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, sar):
        
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0, (self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M, M) + W + self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update C
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - self.lamb - torch.mul(self.beta, H) - L2, self.ones - torch.mul(self.ones, self.beta))

        # Update M

        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update G
        G_k = self.Proximal_G(torch.cat((sar, X_k + L1 / (self.alpha + 1e-6)), dim = 1))

        # Update H
        H_k = self.Proximal_H(C_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_SAR_Trans(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_3_SAR_Trans, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp):
        
        # Y is Y | W is F_att(Y)
        # Update X
        
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M, cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)
        
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - torch.mul(self.ones,self.lamb) + torch.mul(self.beta, H) - L2, self.ones + torch.mul(self.ones, self.beta))

        # Update M
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.5, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_SAR_Trans_2(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_3_SAR_Trans_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp):
        
        # Y is Y | W is F_att(Y)
        # Update X
        
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M, cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)
        
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - torch.mul(self.ones,self.lamb) + torch.mul(self.beta, H) - L2, self.ones + torch.mul(self.ones, self.beta))

        # Update M
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.2, torch.ones_like(C_k), torch.zeros_like(C_k))

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_SAR_Trans_3(nn.Module): # with mask
    def __init__(self):
        super(RPCA_Block_new_3_SAR_Trans_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp):
        
        # Y is Y | W is F_att(Y)
        # Update X
        
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M, cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)
        
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - torch.mul(self.ones,self.lamb) + torch.mul(self.beta, H) - L2, self.ones + torch.mul(self.ones, self.beta))

        # Update M
        # M_k = torch.where(torch.mean(C_k, dim=1) > 0.2, torch.ones_like(C_k), torch.zeros_like(C_k))
        M_k = M

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_3_SAR_Trans_4(nn.Module): # with cloud threshold
    def __init__(self):
        super(RPCA_Block_new_3_SAR_Trans_4, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = UNet_full(in_channels=13, out_channels=13)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)
    
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp):
        
        # Y is Y | W is F_att(Y)
        # Update X
        
        X_k = torch.div(torch.mul(M, Y) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M, cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + torch.mul(self.ones, self.alpha))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)
        
        # C_k = Y - torch.mul(M, X)
        C_k = torch.div(Y - torch.mul(M, X_k) - torch.mul(self.ones,self.lamb) + torch.mul(self.beta, H) - L2, self.ones + torch.mul(self.ones, self.beta))
        # Apply threshold on C_k
        # C_k = torch.where(C_k < 0.2, torch.zeros_like(C_k), C_k)
        # Update M
        M_k = torch.where(torch.mean(C_k, dim=1) > 0.2, torch.ones_like(C_k), torch.zeros_like(C_k))
        C_k = torch.where(C_k < 0.2, torch.zeros_like(C_k), C_k)
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (C_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_noW(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_noW, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + self.tensor_product(L, R) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + self.ones - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_gc(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_gc, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def grabcut_opencv(self, image, mask):
        """ Perform GrabCut using OpenCV for comparison """
        refined = False
        # Convert PyTorch tensors to NumPy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_np[mask_np == 0] = 2
        # print('mask_np shape: ', mask_np.shape)
        # print('mask_np: ', mask_np)
        unique, counts = np.unique(mask_np, return_counts=True)
        percentages = dict(zip(unique, counts / mask_np.size * 100))
        # print("Percentage of values in mask_np:", percentages)
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # rect = (1,1,254,254)
        if 1 < percentages.get(1, 0) < 99:
            
            # Run OpenCV GrabCut
            mask_np, bgdModel, fgdModel = cv2.grabCut(image_np, mask_np, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            refined = True
        # Convert the OpenCV mask to binary (foreground = 1, background = 0)
        opencv_mask = np.where((mask_np == 2) | (mask_np == 0), 0, 1).astype('uint8')
        # unique, counts = np.unique(opencv_mask, return_counts=True)
        # percentages = dict(zip(unique, counts / opencv_mask.size * 100))
        # print("Percentage of values in opencv_mask:", percentages)

        # Apply the mask to the image
        # opencv_segmented = image_np * opencv_mask[:, :, np.newaxis]

        # Convert results back to PyTorch tensors
        opencv_mask_torch = torch.from_numpy(opencv_mask).float()
        # opencv_segmented_torch = torch.from_numpy(opencv_segmented).permute(2, 0, 1).float()

        return opencv_mask_torch, refined #, opencv_segmented_torch

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Refine M_k using GrabCut
        M_k_refined, refined = self.grabcut_opencv(Omega[0, [3,2,1], :, :], M_k[:,0, :, :])
        # print('Changed: ', changed)
        if refined:
            M_k_refined = M_k_refined.repeat(M_k.shape[0], M_k.shape[1], 1, 1).cuda()
            M_k = M_k_refined
            M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_SAR, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB_2(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_gc(nn.Module): # with grabcut
    def __init__(self):
        super(RPCA_Block_new_2_SAR_gc, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB_2(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_2(growRate0=64, growRate=32, nConvLayers=6)
    
    def grabcut_opencv(self, image, mask):
        """ Perform GrabCut using OpenCV for comparison """
        refined = False
        # Convert PyTorch tensors to NumPy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_np[mask_np == 0] = 2
        # print('mask_np shape: ', mask_np.shape)
        # print('mask_np: ', mask_np)
        unique, counts = np.unique(mask_np, return_counts=True)
        percentages = dict(zip(unique, counts / mask_np.size * 100))
        # print("Percentage of values in mask_np:", percentages)
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # rect = (1,1,254,254)
        if 1 < percentages.get(1, 0) < 99:
            
            # Run OpenCV GrabCut
            mask_np, bgdModel, fgdModel = cv2.grabCut(image_np, mask_np, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            refined = True
        # Convert the OpenCV mask to binary (foreground = 1, background = 0)
        opencv_mask = np.where((mask_np == 2) | (mask_np == 0), 0, 1).astype('uint8')
        # unique, counts = np.unique(opencv_mask, return_counts=True)
        # percentages = dict(zip(unique, counts / opencv_mask.size * 100))
        # print("Percentage of values in opencv_mask:", percentages)

        # Apply the mask to the image
        # opencv_segmented = image_np * opencv_mask[:, :, np.newaxis]

        # Convert results back to PyTorch tensors
        opencv_mask_torch = torch.from_numpy(opencv_mask).float()
        # opencv_segmented_torch = torch.from_numpy(opencv_segmented).permute(2, 0, 1).float()

        return opencv_mask_torch, refined #, opencv_segmented_torch
    
    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # Refine M_k using GrabCut
        M_k_refined, refined = self.grabcut_opencv(Omega[0, [3,2,1], :, :], M_k[:,0, :, :])
        # print('Changed: ', changed)
        if refined:
            M_k_refined = M_k_refined.repeat(M_k.shape[0], M_k.shape[1], 1, 1).cuda()
            M_k = M_k_refined
            M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_gc_torch(nn.Module): # with torchmaxflow
    def __init__(self):
        super(RPCA_Block_new_2_SAR_gc_torch, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB_2(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_2(growRate0=64, growRate=32, nConvLayers=6)
    
    def grabcut_opencv(self, image, mask):
        """ Perform GrabCut using OpenCV for comparison """
        refined = False
        # Convert PyTorch tensors to NumPy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_np[mask_np == 0] = 2
        # print('mask_np shape: ', mask_np.shape)
        # print('mask_np: ', mask_np)
        unique, counts = np.unique(mask_np, return_counts=True)
        percentages = dict(zip(unique, counts / mask_np.size * 100))
        # print("Percentage of values in mask_np:", percentages)
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # rect = (1,1,254,254)
        if 1 < percentages.get(1, 0) < 99:
            
            # Run OpenCV GrabCut
            mask_np, bgdModel, fgdModel = cv2.grabCut(image_np, mask_np, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            refined = True
        # Convert the OpenCV mask to binary (foreground = 1, background = 0)
        opencv_mask = np.where((mask_np == 2) | (mask_np == 0), 0, 1).astype('uint8')
        # unique, counts = np.unique(opencv_mask, return_counts=True)
        # percentages = dict(zip(unique, counts / opencv_mask.size * 100))
        # print("Percentage of values in opencv_mask:", percentages)

        # Apply the mask to the image
        # opencv_segmented = image_np * opencv_mask[:, :, np.newaxis]

        # Convert results back to PyTorch tensors
        opencv_mask_torch = torch.from_numpy(opencv_mask).float()
        # opencv_segmented_torch = torch.from_numpy(opencv_segmented).permute(2, 0, 1).float()

        return opencv_mask_torch, refined #, opencv_segmented_torch
    
    def graphcut_pytorchmaxflow(self, image, mask):
        
        fP = 0.5 + (mask - 0.5) * 0.8
        bP = 1.0 - fP
        Prob = torch.cat([bP, fP], dim=1)

        img = torch.clone(image)
        img = img.detach().cpu()
        Prob = Prob.detach().cpu()
        lamda = 20.0
        sigma = 10.0
        post_proc_label = torchmaxflow.maxflow(img, Prob, lamda, sigma, 4)

        return post_proc_label.to(image.device)


    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # M_k = torch.clamp(M_k, min=0.0, max=1.0)
        # Refine M_k using GrabCut
        # M_k_refined, refined = self.grabcut_opencv(Omega[0, [3,2,1], :, :], M_k[:,0, :, :])
        M_k_refined = self.graphcut_pytorchmaxflow(Omega[:, [3,2,1], :, :], M_k[:,0:1, :, :])
        # print('Changed: ', changed)
        # if refined:
            # M_k_refined = M_k_refined.repeat(M_k.shape[0], M_k.shape[1], 1, 1).cuda()
        M_k = M_k_refined
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_CS(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_SAR_CS, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * cs_comp - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_CS_noW(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_SAR_CS_noW, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, cs_comp):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + self.tensor_product(L, R) - self.gamma * cs_comp - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + self.ones - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_CS_2(nn.Module): # RPCA_Block_new_2_SAR_CS_2 with grab cut
    def __init__(self):
        super(RPCA_Block_new_2_SAR_CS_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)
    
    def grabcut_opencv(self, image, mask):
        """ Perform GrabCut using OpenCV for comparison """
        refined = False
        # Convert PyTorch tensors to NumPy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_np[mask_np == 0] = 2
        # print('mask_np shape: ', mask_np.shape)
        # print('mask_np: ', mask_np)
        unique, counts = np.unique(mask_np, return_counts=True)
        percentages = dict(zip(unique, counts / mask_np.size * 100))
        # print("Percentage of values in mask_np:", percentages)
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # rect = (1,1,254,254)
        if 1 < percentages.get(1, 0) < 99:
            
            # Run OpenCV GrabCut
            mask_np, bgdModel, fgdModel = cv2.grabCut(image_np, mask_np, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            refined = True
        # Convert the OpenCV mask to binary (foreground = 1, background = 0)
        opencv_mask = np.where((mask_np == 2) | (mask_np == 0), 0, 1).astype('uint8')
        # unique, counts = np.unique(opencv_mask, return_counts=True)
        # percentages = dict(zip(unique, counts / opencv_mask.size * 100))
        # print("Percentage of values in opencv_mask:", percentages)

        # Apply the mask to the image
        # opencv_segmented = image_np * opencv_mask[:, :, np.newaxis]

        # Convert results back to PyTorch tensors
        opencv_mask_torch = torch.from_numpy(opencv_mask).float()
        # opencv_segmented_torch = torch.from_numpy(opencv_segmented).permute(2, 0, 1).float()

        return opencv_mask_torch, refined#, opencv_segmented_torch

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * cs_comp - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Refine M_k using GrabCut
        M_k_refined, refined = self.grabcut_opencv(Omega[0, [3,2,1], :, :], M_k[:,0, :, :])
        # print('Changed: ', changed)
        if refined:
            M_k_refined = M_k_refined.repeat(M_k.shape[0], M_k.shape[1], 1, 1).cuda()
            M_k = M_k_refined
            M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # print(M_k)
        # print('M_k shape: ', M_k.shape)
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k
    
class RPCA_Block_new_2_SAR_CS_3(nn.Module): # RPCA_Block_new_2_SAR_CS_2 with graph cut pytorchmaxflow
    def __init__(self):
        super(RPCA_Block_new_2_SAR_CS_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)
    
    def grabcut_opencv(self, image, mask):
        """ Perform GrabCut using OpenCV for comparison """
        refined = False
        # Convert PyTorch tensors to NumPy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_np[mask_np == 0] = 2
        # print('mask_np shape: ', mask_np.shape)
        # print('mask_np: ', mask_np)
        unique, counts = np.unique(mask_np, return_counts=True)
        percentages = dict(zip(unique, counts / mask_np.size * 100))
        # print("Percentage of values in mask_np:", percentages)
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # rect = (1,1,254,254)
        if 1 < percentages.get(1, 0) < 99:
            
            # Run OpenCV GrabCut
            mask_np, bgdModel, fgdModel = cv2.grabCut(image_np, mask_np, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            refined = True
        # Convert the OpenCV mask to binary (foreground = 1, background = 0)
        opencv_mask = np.where((mask_np == 2) | (mask_np == 0), 0, 1).astype('uint8')
        # unique, counts = np.unique(opencv_mask, return_counts=True)
        # percentages = dict(zip(unique, counts / opencv_mask.size * 100))
        # print("Percentage of values in opencv_mask:", percentages)

        # Apply the mask to the image
        # opencv_segmented = image_np * opencv_mask[:, :, np.newaxis]

        # Convert results back to PyTorch tensors
        opencv_mask_torch = torch.from_numpy(opencv_mask).float()
        # opencv_segmented_torch = torch.from_numpy(opencv_segmented).permute(2, 0, 1).float()

        return opencv_mask_torch, refined#, opencv_segmented_torch

    def graphcut_pytorchmaxflow(self, image, mask):
        
        fP = 0.5 + (mask - 0.5) * 0.8
        bP = 1.0 - fP
        Prob = torch.cat([bP, fP], dim=1)

        img = torch.clone(image)
        img = img.detach().cpu()
        Prob = Prob.detach().cpu()
        lamda = 20.0
        sigma = 10.0
        post_proc_label = torchmaxflow.maxflow(img, Prob, lamda, sigma, 4)

        return post_proc_label.to(image.device)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp):
        
        # Omega is Y | W is F_att(Y)
        # Update X
        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * cs_comp - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1,\
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # M_k = torch.clamp(M_k, min=0.0, max=1.0)
        # Refine M_k using GrabCut
        # M_k_refined, refined = self.grabcut_opencv(Omega[0, [3,2,1], :, :], M_k[:,0, :, :])
        M_k_refined = self.graphcut_pytorchmaxflow(Omega[:, [3,2,1], :, :], M_k[:,0:1, :, :])
        # print('Changed: ', changed)
        # if refined:
            # M_k_refined = M_k_refined.repeat(M_k.shape[0], M_k.shape[1], 1, 1).cuda()
        M_k = M_k_refined
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # print(M_k)
        # print('M_k shape: ', M_k.shape)
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_Trans(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M , cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.gamma * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_Trans_2(nn.Module): # Stronger contraints on trans
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi | L3 is Alpha
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M , cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1 \
                        - torch.mul(L3,M), torch.mul(M,M) + W + self.gamma * M - self.mu * (self.ones - M) + self.alpha)
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        L3_k = L3 + self.gamma * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k

class RPCA_Block_new_2_SAR_Trans_3(nn.Module): #multi batchsize
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + self.gamma * torch.mul(M , cs_comp) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.gamma + self.alpha)
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.gamma * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_new_2_SAR_Trans_3_0(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_3_0, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + self.gamma * torch.mul(M , cs_comp) - \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.gamma + self.alpha)
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.gamma * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, self.lamb, self.gamma, self.mu, self.alpha, self.beta

class RPCA_Block_new_2_SAR_Trans_3_0_0(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_3_0_0, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + torch.abs(self.gamma) * torch.mul(M , cs_comp) - \
                        torch.abs(self.mu) * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + torch.abs(self.alpha) * G - L1, \
                        torch.mul(M,M) + W - torch.abs(self.mu) * (self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - torch.abs(self.lamb) + \
                        torch.abs(self.mu) * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + torch.abs(self.beta) * H - L2,\
                        torch.mul(X,X)+ torch.abs(self.mu) * torch.mul(Omega - X, Omega - X) + torch.abs(self.beta))

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_new_2_SAR_Trans_3_0_0(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_3_0_0, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + torch.abs(self.gamma) * torch.mul(M , cs_comp) - \
                        torch.abs(self.mu) * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + torch.abs(self.alpha) * G - L1, \
                        torch.mul(M,M) + W - torch.abs(self.mu) * (self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - torch.abs(self.lamb) + \
                        torch.abs(self.mu) * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + torch.abs(self.beta) * H - L2,\
                        torch.mul(X,X)+ torch.abs(self.mu) * torch.mul(Omega - X, Omega - X) + torch.abs(self.beta))

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)


class RPCA_Block_new_2_SAR_Trans_3_1(nn.Module): #multi batchsize + check parameters + Mask Refinement
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_3_1, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)
        
    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) + torch.abs(self.gamma) * torch.mul(M , cs_comp) - \
                        torch.abs(self.mu) * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + torch.abs(self.alpha) * G - L1, \
                        torch.mul(M,M) + W - torch.abs(self.mu) * (self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - torch.abs(self.lamb) + \
                        torch.abs(self.mu) * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + torch.abs(self.beta) * H - L2,\
                        torch.mul(X,X)+ torch.abs(self.mu) * torch.mul(Omega - X, Omega - X) + torch.abs(self.beta))
        
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))

        # Create trimap
        M_k = M_k[0,0,...]
        trimap = torch.zeros_like(M_k)
        trimap[M_k > 0.5] = 1
        trimap[M_k < 0.5] = 0
        uncertain_fg = M_k.clone()
        uncertain_fg = F.conv2d(uncertain_fg.unsqueeze(0).unsqueeze(0).float(), 
                                weight=torch.ones(1, 1, 51, 51, device=uncertain_fg.device), 
                                padding=25).squeeze(0).squeeze(0) > 0
        # print("uncertain_fg.shape", uncertain_fg.shape)
        # print("trimap.shape", trimap.shape)
        trimap[uncertain_fg == 1] = 0.5

        # Mask Refinement
        # Assuming Omega is the input image and M_k is the mask
        kernel_radius = 20     # Larger radius as suggested by the paper (e.g., 20 for Fig 1d)
        epsilon_reg = 1e-6   # Regularization for matrix inversion stability
        lambda_constraint = 100.0 # Weight for trimap constraints
        cg_tolerance = 1e-5   # Convergence tolerance for CG
        cg_max_iterations = 200 # Max iterations for CG (might need fewer with large kernel)
        compute_device = 'cuda' # Use 'cuda' if available, otherwise 'cpu'

        solver = FastMattingSolver(Omega[0,[3,2,1],...], trimap.unsqueeze(0), radius=kernel_radius,
                               epsilon=epsilon_reg, lambda_val=lambda_constraint,
                               device=compute_device)

        alpha_matte = solver.solve(tol=cg_tolerance, max_iter=cg_max_iterations)
        M_k = (alpha_matte > 0.1).float()
        M_k = M_k.repeat(13,1,1)
        M_k = M_k.unsqueeze(0)
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_new_2_SAR_Trans_torch(nn.Module):
    def __init__(self):
        super(RPCA_Block_new_2_SAR_Trans_torch, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, X_k):
        X = torch.fft.fft(torch.squeeze(X_k), n=X_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(X, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), X).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Omega, W, cs_comp): # Omega is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Omega is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Omega) - torch.mul(M, C) + torch.mul(W, self.tensor_product(L, R)) - self.gamma * torch.mul(M , cs_comp) + \
                        self.mu * torch.mul((self.ones - M), Omega) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W - self.mu * (self.ones - M) + self.alpha)
        
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M
        M_k = torch.div(torch.mul(Omega,X) - torch.mul(C, X) - self.lamb + \
                        self.mu * torch.mul(Omega - X,Omega - X) + torch.mul(L0,(Omega-X)) + self.beta * H - L2,\
                        torch.mul(X,X)+ self.mu * torch.mul(Omega - X, Omega - X) + self.beta)

        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        M_k = torch.where(torch.mean(M_k, dim=1) > 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Omega - torch.mul(M, X)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (self.alpha + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (self.beta + 1e-6))

        # Update Lambda
        L0_k = L0 + self.mu * torch.mul((self.ones - M_k), Omega - X_k)
        L1_k = L1 + self.alpha * (X_k - G_k)
        L2_k = L2 + self.beta * (M_k - H_k)
        # L3_k = L3 + self.gamma * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k

class RPCA_Block_4_SAR_Trans_1(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_1, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = Y - torch.mul(M_k, X_k)

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)

        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_4_SAR_Trans_2(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Make sure M_k is in the range [0, 1] by using sigmoid
        M_k = torch.sigmoid(M_k)
        
        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_4_SAR_Trans_3(nn.Module): #multi batchsize + check parameters
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_3, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Make sure M_k is in the range [0, 1] by using sigmoid
        # M_k = torch.sigmoid(M_k)
        
        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_4_SAR_Trans_4(nn.Module): #multi batchsize + check parameters + quantize M and C
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_4, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Make sure M_k is in the range [0, 1] by using sigmoid
        M_k = torch.sigmoid(M_k)
        
        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_4_SAR_Trans_5(nn.Module): #multi batchsize + check parameters + quantize M and C
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_5, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Make sure M_k is in the range [0, 1] by scaling
        M_k = torch.relu(M_k)
        M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        # Make sure C_k is in the range [0, 1] by scaling
        C_k = torch.relu(C_k)
        C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta)

class RPCA_Block_4_SAR_Trans_6(nn.Module): #multi batchsize + check parameters + KKT on M (no clamping M)
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_6, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho_U = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.rho_V = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, U, V, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2 + U - V, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update U and V

        U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), U_k, V_k

class RPCA_Block_4_SAR_Trans_6_2(nn.Module): #multi batchsize + check parameters + KKT on M (clamping M)
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_6_2, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho_U = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.rho_V = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, U, V, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2 + U - V, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update U and V
        
        U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        # L3_k = L3 + torch.abs(self.gamma) * (X_k - cs_comp)
        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), U_k, V_k


class RPCA_Block_4_SAR_Trans_7(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_7, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + torch.mul(L0,(self.ones - M)) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.mul(L0, Y - X_k) + torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k


class RPCA_Block_4_SAR_Trans_8(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_unetresprox(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_unetresprox, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = UNetRes(in_nc=13, out_nc=13, nc=[64, 128, 256, 512], nb=2)
        self.Proximal_Q = UNetRes(in_nc=13, out_nc=1, nc=[64, 128, 256, 512], nb=2)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_convreluprox(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_convreluprox, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = ProximalBlock(out_channels=13)
        self.Proximal_Q = ProximalBlock(out_channels=1)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_wo_att(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_wo_att, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + self.tensor_product(L, R) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + self.ones + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_wo_S(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_wo_S, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) +\
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_wo_DI(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_wo_DI, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k


class RPCA_Block_4_SAR_Trans_8_wo_LR(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_wo_LR, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        # L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L, R, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_8_wo_R(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_8_wo_R, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) , \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) \
                        + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        # G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        # H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        # L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        # L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G, H, L0_k, L1, L2, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_9(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_9, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, V, T, Y, W, X_hat): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi | L3 is Lambda_V | L4 is zeta
        # Y is Y | W is F_att(Y)
        # Update X

        # X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
        #                 torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
        #                 torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        
        one_minus_M = self.ones - M
        one_minus_M_sq = torch.mul(one_minus_M, one_minus_M)
        Psi_k = one_minus_M * (Y - M * C) + \
                self.mu * one_minus_M_sq * Y + \
                self.gamma * X_hat + \
                self.alpha * G - \
                L1
        
        numerator = torch.mul(W, self.tensor_product(L, R)) + Psi_k
        denominator = (1+self.mu) * one_minus_M_sq + W + torch.abs(self.gamma) + torch.abs(self.alpha)
        X_k = torch.div(numerator, denominator)
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        # M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
        #                 torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
        #                 torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        # M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        X_minus_C = X_k - C
        X_minus_Y = X_k - Y

        # 1. Compute the quadratic weight tensor A_k
        A_k = torch.mul(X_minus_C, X_minus_C) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + torch.abs(self.beta) + torch.abs(self.eta) + torch.abs(self.rho)

        # 2. Compute the linear weight tensor Upsilon_k
        Upsilon_k = torch.mul(X_minus_C, X_minus_Y) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + \
                    torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * V - L3 + torch.abs(self.eta) * (self.ones - T) - L4

        # 3. Update M_k using element-wise division
        X_input = torch.div(Upsilon_k, A_k + 1e-6)  # Add a small constant to avoid division by zero
        tau = self.lamb / (A_k + 1e-6)  # Compute the threshold for soft-thresholding

        # 4. Apply element-wise Soft-Thresholding: sign(x) * max(|x| - tau, 0)
        
        M_k = torch.sign(X_input) * nn.functional.relu(torch.abs(X_input) - tau)
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]

        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
            print("X_k is nan")
            print("M_k is nan")
            print("C_k is nan")
            a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k


class RPCA_Block_4_SAR_Trans_smilecr_8(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_smilecr_8, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_smilecr(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2_smilecr(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_smilecr_8_wo_LR(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_smilecr_8_wo_LR, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_smilecr(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2_smilecr(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        # L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L, R, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_smilecr_8_wo_R(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_smilecr_8_wo_R, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_smilecr(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2_smilecr(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + torch.mul(torch.abs(self.gamma), cs_comp) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) , \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.gamma))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) \
                        + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        # G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        # H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        # L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        # L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G, H, L0_k, L1, L2, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class RPCA_Block_4_SAR_Trans_smilecr_8_woDI(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_4_SAR_Trans_smilecr_8_woDI, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_smilecr(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2_smilecr(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, Y, W, cs_comp): # Y is Y | W is F_att(Y)
        
        # L0 is Lambda | L1 is Gamma | L2 is Phi 
        # Y is Y | W is F_att(Y)
        # Update X

        X_k = torch.div(torch.mul(M, Y - torch.mul(C, (self.ones-M))) + torch.mul(W, self.tensor_product(L, R)) + \
                        torch.abs(self.mu) * torch.mul(torch.mul(self.ones - M, self.ones - M), Y) + self.alpha * G - L1, \
                        torch.mul(M,M) + W + torch.abs(self.mu) * torch.mul(self.ones - M, self.ones - M) + torch.abs(self.alpha))
        # print("mean(X_k)", torch.mean(X_k))
        # print("max(X_k)", torch.max(X_k))
        # print("mean(Y)", torch.mean(Y))
        # print("max(Y)", torch.max(Y))
        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        M_k = torch.div(torch.mul(X_k - C, Y - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) - torch.abs(self.lamb) + \
                        torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * S - L3 + torch.abs(self.eta) * (self.ones - T) - L4, \
                        torch.mul(X_k - C, X_k - C) + torch.abs(self.mu) * torch.mul(Y - X_k, Y - X_k) + torch.abs(self.beta) + torch.abs(self.rho) + torch.abs(self.eta))
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
        # Make sure M_k is in the range [0, 1] by scaling
        # M_k = torch.relu(M_k)
        # M_k = torch.div(M_k, torch.max(M_k) + 1e-6)  # Normalize M_k to [0, 1]
        # M_k = torch.sigmoid(M_k)
        
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
        # U_k = torch.clamp(U - self.rho_U * M_k, min=0)  # Ensure U_k >= 0
        # V_k = torch.clamp(V - self.rho_V * (M_k - self.ones), min=0)  # Ensure V_k >= 0

        # print("mean(M_k)", torch.mean(M_k))
        # print("max(M_k)", torch.max(M_k))
        # Threshold M_k
        # M_k = torch.where(M_k >= 0.5, torch.ones_like(M_k), torch.zeros_like(M_k))
        # Synchronize channels
        # M_k = torch.where(torch.mean(M_k, dim=1, keepdim = True) > 0.1, torch.ones_like(M_k), torch.zeros_like(M_k))
        
        # Update C
        C_k = torch.div(Y - torch.mul(M_k, X_k) , (self.ones - M_k)+ 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]
        # Make sure C_k is in the range [0, 1] by scaling
        # C_k = torch.relu(C_k)
        # C_k = torch.div(C_k, torch.max(C_k) + 1e-6)  # Normalize C_k to [0, 1]
        # C_k = torch.sigmoid(C_k)  # Quantize C_k to [0, 1]
        # C_k = Y - torch.mul(M_k, X_k)
        # print("mean(C_k)", torch.mean(C_k))
        # print("max(C_k)", torch.max(C_k))
        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        # L0_k = L0 + torch.abs(self.mu) * torch.mul((self.ones - M_k), Y - X_k)
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # a = 1/0
        # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any():
        #     print("X_k is nan")
        # if torch.isnan(M_k).any():
        #     print("M_k is nan")
        # if torch.isnan(C_k).any():
        #     print("C_k is nan")
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k


class RPCA_Net_new_2(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return X
    
class RPCA_Net_new_2_stages_cpu(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_cpu, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size()) #, device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size()) #, device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size()) #, device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size()) #, device=torch.device('cuda')) # M
        C  = torch.zeros(X.size()) #, device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size()) #, device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size()) #, device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1])) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1])) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_ver2(nn.Module): # with U-Net
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_ver2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_ver2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_ver2_1(nn.Module): # with U-Net multi batchsize
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_ver2_1, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_ver2_1())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((OmegaD.shape[0], 13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((OmegaD.shape[0], 10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers

class RPCA_Net_new_3_stages(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, Y, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers


class RPCA_Net_new_3_stages_ver2(nn.Module): # with U-Net
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, Y, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_3_stages_ver3(nn.Module): # with U-Net
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers
    
class RPCA_Net_new_3_stages_ver3_2(nn.Module): # with U-Net
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers

class RPCA_Net_new_3_stages_ver3_3(nn.Module): # with Mask
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3_3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3_3())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data, mask):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        # M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        M = mask
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers

class RPCA_Net_new_3_stages_ver3_SAR(nn.Module): # with U-Net
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3_SAR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3_SAR())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        Y = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # Y = torch.mul(Y, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((15, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers

class RPCA_Net_new_3_stages_ver3_SAR_CNN(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3_SAR_CNN, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        Y = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])

        return layers

class RPCA_Net_new_3_stages_ver3_SAR_CNN_2(nn.Module): # Different Unet input
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_ver3_SAR_CNN_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        # self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_ver3_SAR_CS_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        Y = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules

        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        # output = self.conv3(rs)  # Bsx13x64x64
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(Y)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = Y
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, Y.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, Y.shape[-2], Y.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, Y, W, SAR)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])

        return layers

##################################################
##### 3 stages with SAR and Transmittance ########
##################################################

class RPCA_Net_new_3_stages_SAR_Trans_RGB(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_SAR_Trans_RGB, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_SAR_Trans())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])

        return layers

class RPCA_Net_new_3_stages_SAR_Trans_RGB_2(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_SAR_Trans_RGB_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_SAR_Trans_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])

        return layers

class RPCA_Net_new_3_stages_SAR_Trans_RGB_3(nn.Module): # with Mask
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_SAR_Trans_RGB_3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_SAR_Trans_3())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans, mask):

        # Observation
        OmegaD = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        # M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        M  = mask
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])

        return layers

class RPCA_Net_new_3_stages_SAR_Trans_RGB_4(nn.Module): # with cloud threshold
    def __init__(self, N_iter):
        super(RPCA_Net_new_3_stages_SAR_Trans_RGB_4, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_3_SAR_Trans_4())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])

        return layers

##################################################
####### 2 stages with SAR and Transmittance ######
##################################################

class RPCA_Net_new_2_stages_noW(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_noW, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        # self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_noW())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        # W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_W(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_W, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, W])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers
    
class RPCA_Net_new_2_stages_gc(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_gc, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_gc())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_sharedw(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_sharedw, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        # blocks_list = []
        # for i in range(self.N_iter):
        #     blocks_list.append(RPCA_Block_new_2())
        # self.network = nn.ModuleList(blocks_list)
        self.network = RPCA_Block_new_2()

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        # for i in range(0, self.N_iter):
        #     [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
        #     layers.append([L, R, X, M, C, G, H, L0, L1, L2])
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network(L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((15, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers
    
class RPCA_Net_new_2_stages_SAR_gc(nn.Module): # with graph cut
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_gc, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_gc())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((15, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_gc_torch(nn.Module): # with graph cut torchmaxflow
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_gc_torch, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_gc_torch())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((15, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CS(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CS, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        Il = self.I_l_conv(RSI)   
        Gi = self.att_module2(torch.cat((RSI, SAR, Il), dim=1))    
        P_Il = self.dup_conv(SAR - torch.Tensor.repeat(Il,(1,SAR.shape[1],1,1)))
        cs_comp = RSI + torch.mul(Gi, P_Il)

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_MRA(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_MRA, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.PL_conv1 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.PL_conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        PL = self.PL_conv2(self.relu(self.PL_conv1(SAR)))
        Gi = self.att_module2(torch.cat((RSI, SAR, PL), dim=1))    
        P_PL = self.dup_conv(SAR - torch.Tensor.repeat(PL,(1, SAR.shape[1], 1, 1)))
        cs_comp = RSI + torch.mul(Gi, P_PL)

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

"""
class RPCA_Net_new_2_stages_SAR_CNN(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers
"""

class RPCA_Net_new_2_stages_SAR_CNN(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN_noW(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN_noW, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        # self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS_noW())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        # W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN_W(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN_W, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, W])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN_gc(nn.Module): # with grabcut
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN_gc, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):           
            blocks_list.append(RPCA_Block_new_2_SAR_CS_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN_gc_torch(nn.Module): # with torchmaxflow
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN_gc_torch, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):           
            blocks_list.append(RPCA_Block_new_2_SAR_CS_3())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        P_Il = self.dup_conv(SAR)
        rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN2(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_CNN3(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_CNN3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        self.unet = nn.ModuleDict({
            'encoder': nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
            ),
            'decoder': nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
            )
        })

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_CS())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        # rs = self.relu(self.conv1())  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        # output = self.conv3(rs)  # Bsx13x64x64
        # cs_comp = RSI + output
        # Encoder
        skip1 = self.unet['encoder'][0:2](torch.cat((RSI, SAR), 1))  # First conv + ReLU
        skip2 = self.unet['encoder'][2:4](skip1)  # Second conv + ReLU
        encoded = self.unet['encoder'][4:](skip2)  # Third conv + ReLU

        # Decoder with skip connections
        decoded = self.unet['decoder'][0](encoded) + skip2  # First transpose conv + skip2
        decoded = self.unet['decoder'][1](decoded) + skip1  # Second transpose conv + skip1
        output = self.unet['decoder'][2](decoded)  # Final conv
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_2(nn.Module): # stronger constraint
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans_2())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        # print(f"L3 shape: {L3.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_3(nn.Module): #multi-batchsize
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans_3())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((OmegaD.shape[0], 13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((OmegaD.shape[0], 10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])
        # for i in range(0, self.N_iter):
        #     [L, R, X, M, C, G, H, L0, L1, L2] = self.network(L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
        #     layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])
        # parameters = [0.0182, 0.0364, 0.0545, 0.0727, 0.0909, 0.1091, 0.1273, 0.1455, 0.1636, 0.1818]
        # total_loss = sum(parameters[j] * self.loss_fn(layers[j][2],  ground_truth) for j in range(len(layers)))
        
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_3_0(nn.Module): # multi-batchsize + check the parameters
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_3_0, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans_3_0())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((OmegaD.shape[0], 13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((OmegaD.shape[0], 10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta])
        # for i in range(0, self.N_iter):
        #     [L, R, X, M, C, G, H, L0, L1, L2] = self.network(L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
        #     layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])
        # parameters = [0.0182, 0.0364, 0.0545, 0.0727, 0.0909, 0.1091, 0.1273, 0.1455, 0.1636, 0.1818]
        # total_loss = sum(parameters[j] * self.loss_fn(layers[j][2],  ground_truth) for j in range(len(layers)))
        
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_3_1(nn.Module): # multi-batchsize + check the parameters + Mask Refinement
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_3_1, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans_3_1())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((OmegaD.shape[0], 13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((OmegaD.shape[0], 10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta])
        # for i in range(0, self.N_iter):
        #     [L, R, X, M, C, G, H, L0, L1, L2] = self.network(L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
        #     layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])
        # parameters = [0.0182, 0.0364, 0.0545, 0.0727, 0.0909, 0.1091, 0.1273, 0.1455, 0.1636, 0.1818]
        # total_loss = sum(parameters[j] * self.loss_fn(layers[j][2],  ground_truth) for j in range(len(layers)))
        
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_3_0_0(nn.Module): # multi-batchsize + check the parameters + abs parameters
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_3_0_0, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans_3_0_0())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((OmegaD.shape[0], 13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((OmegaD.shape[0], 10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta])
        # for i in range(0, self.N_iter):
        #     [L, R, X, M, C, G, H, L0, L1, L2] = self.network(L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
        #     layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])
        # parameters = [0.0182, 0.0364, 0.0545, 0.0727, 0.0909, 0.1091, 0.1273, 0.1455, 0.1636, 0.1818]
        # total_loss = sum(parameters[j] * self.loss_fn(layers[j][2],  ground_truth) for j in range(len(layers)))
        
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_dup(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_dup, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.att_module2 = SpatialAttentionModule_SAR(13+2+1)
        # self.I_l_conv = nn.Conv2d(13, 1, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        # self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        # CS modules
        # P_Il = self.dup_conv(SAR)
        # rs = self.relu(self.conv1(P_Il - RSI))  # Bsx32x64x64
        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        # cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        trans_avg = torch.mean(trans, dim=1, keepdim=True)
        cs_comp = torch.cat((trans,trans,trans,trans,trans_avg), 1)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_new_2_stages_SAR_Trans_RGB_dup_torch(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_2_stages_SAR_Trans_RGB_dup_torch, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new_2_SAR_Trans())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        OmegaD = RSI
        # Omega_C = torch.tensor(1.).cuda() -  omega
        
        trans_avg = torch.mean(trans, dim=1, keepdim=True)
        cs_comp = torch.cat((trans,trans,trans,trans,trans_avg), 1)  # Bsx13x256x256
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        # L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Alpha
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, OmegaD, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp])


            # [L, R, C, E, P, Q, L0, L1, L2] = self.network[i](L, R, C, E, P, Q, L0, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L0 shape: {L0.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_1(nn.Module): # multi-batchsize + check the parameters + abs parameters
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_1, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_1())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, L0, L1, L2])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_2(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_2())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, L0, L1, L2])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_3(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_3())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, L0, L1, L2])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_4(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule + quantize M and C
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_4, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_4())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, L0, L1, L2])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_5(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule + scale M and C
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_5, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_5())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        
        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, L0, L1, L2])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_6(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule + KKT condition M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_6, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_6())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        U = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        V = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta, U, V] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, U, V, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, U, V])

        return layers


class RPCA_Net_4_stages_SAR_Trans_RGB_6_2(nn.Module): # multi-batchsize + check the parameters + abs parameters + new C_k updating rule + KKT condition M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_6_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_6_2())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        U = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        V = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, lamb, gamma, mu, alpha, beta, U, V] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, U, V, X, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, cs_comp, lamb, gamma, mu, alpha, beta, U, V])

        return layers


class RPCA_Net_4_stages_SAR_Trans_RGB_7(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_7, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_7())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers


class RPCA_Net_4_stages_SAR_Trans_RGB_8(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_CNN(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_CNN, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        # self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        # cs_comp = self.dup_conv(trans)  # Bsx13x256x256
        rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx13x64x64
        cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers
    
class RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_CNN_0(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_CNN_0, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(SAR)  # Bsx13x256x256
        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        # output = self.conv3(rs)  # Bsx13x64x64
        # cs_comp = RSI + output

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_dub(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_SAR_dub, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        # self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     self.res1,
        #     self.res2,
        #     self.res3,
        #     self.res4
        # )
        # self.dup_conv = nn.Conv2d(2, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        # cs_comp = self.dup_conv(SAR)  # Bsx13x256x256
        # rs = self.relu(self.conv1(torch.cat((RSI, SAR), 1)))  # Bsx32x64x64
        # rs = self.backbone(rs)  # ResNet's backbone!
        # output = self.conv3(rs)  # Bsx13x64x64
        # cs_comp = RSI + output
        # Duplicate SAR from 2 to 13 channels
        # cs_comp = SAR.repeat(1,13,1,1)  # Bsx13x256x256
        cs_comp = torch.cat([SAR, SAR, SAR, SAR, SAR, SAR, SAR[:,0,:,:].unsqueeze(1)], dim=1)

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_convreluprox(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_convreluprox, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_convreluprox())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_unetresprox(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_unetresprox, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_unetresprox())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_simam(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_simam, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = simam_module(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_PALayer(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_PALayer, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = PALayer(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_SAM(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_SAM, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SAM()
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_att(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_att, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        # self.att_module = SAM()
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_wo_att())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = 0

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers
    
class RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_S(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_S, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_wo_S())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_DI(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_DI, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_wo_DI())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers


class RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_LR(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_LR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_wo_LR())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers


class RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_R(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_8_wo_R, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_8_wo_R())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(6)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_smilecr_8())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 6, 4, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 4, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_woDI(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_woDI, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(6)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_smilecr_8_woDI())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 6, 4, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 4, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_wo_R(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_wo_R, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(6)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_smilecr_8_wo_R())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 6, 4, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 4, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_wo_LR(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_SMILECR_8_wo_LR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(6)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_smilecr_8_wo_LR())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 6, 4, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 4, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Net_4_stages_SAR_Trans_RGB_9(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(RPCA_Net_4_stages_SAR_Trans_RGB_9, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_4_SAR_Trans_9())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        V = torch.zeros(X.size(), device=torch.device('cuda')) # V for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # T for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, V, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, V, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, V, T])

        return layers


class RPCA_Net_new(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L1, L2] = self.network[i](L, R, X, M, C, G, H, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L1, L2])


            # [L, R, C, E, P, Q, L1, L2] = self.network[i](L, R, C, E, P, Q, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        # print(f"L shape: {L.shape}")
        # print(f"R shape: {R.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"M shape: {M.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"G shape: {G.shape}")
        # print(f"H shape: {H.shape}")
        # print(f"L1 shape: {L1.shape}")
        # print(f"L2 shape: {L2.shape}")
        return X

class RPCA_Net_new_stage(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_new_stage, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_new())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data):

        # Observation
        OmegaD = data
        # Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        # OmegaD = torch.mul(OmegaD, omega)
        X  = OmegaD
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        
        M  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L1, L2] = self.network[i](L, R, X, M, C, G, H, L1, L2, OmegaD, W)
            layers.append([L, R, X, M, C, G, H, L1, L2])


            # [L, R, C, E, P, Q, L1, L2] = self.network[i](L, R, C, E, P, Q, L1, L2, OmegaD, W)
            # layers.append([L, R, C, E, P, Q, L1, L2])
    
        # L, R, X, M, C, G, H, L1, L2 = layers[-1]
        
        return layers

class RPCA_Net(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data, omega):

        # Observation
        OmegaD = data
        Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        OmegaD = torch.mul(OmegaD, omega)
        C  = OmegaD
        L1 = torch.zeros(C.size(), device=torch.device('cuda')) #Lambda
        L2 = torch.zeros(C.size(), device=torch.device('cuda')) #Gamma
        L3 = torch.zeros(C.size(), device=torch.device('cuda'))
        E  = torch.zeros(C.size(), device=torch.device('cuda')) # C
        T  = torch.zeros(C.size(), device=torch.device('cuda')) # skip
        P  = torch.zeros(C.size(), device=torch.device('cuda')) # G for X
        Q  = torch.zeros(C.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((13, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, C, E, T, P, Q, L1, L2, L3] = self.network[i](L, R, C, E, T, P, Q, L1, L2, L3, OmegaD, W, Omega_C)
            layers.append([L, R, C, E, T, P, Q, L1, L2, L3])

        return C

class RPCA_Net_R3(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_R3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, data, omega):

        # Observation
        OmegaD = data
        Omega_C = torch.tensor(1.).cuda() -  omega
        
        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        OmegaD = torch.mul(OmegaD, omega)
        C  = OmegaD
        L1 = torch.zeros(C.size(), device=torch.device('cuda'))
        L2 = torch.zeros(C.size(), device=torch.device('cuda'))
        L3 = torch.zeros(C.size(), device=torch.device('cuda'))
        E  = torch.zeros(C.size(), device=torch.device('cuda'))
        T  = torch.zeros(C.size(), device=torch.device('cuda'))
        P  = torch.zeros(C.size(), device=torch.device('cuda'))
        Q  = torch.zeros(C.size(), device=torch.device('cuda'))
        # Init L/R
        L = torch.ones((13, 3, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((3, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, C, E, T, P, Q, L1, L2, L3] = self.network[i](L, R, C, E, T, P, Q, L1, L2, L3, OmegaD, W, Omega_C)
            layers.append([L, R, C, E, T, P, Q, L1, L2, L3])

        return C
    
class RPCA_Net_2(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_2, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_2())
        self.network = nn.ModuleList(blocks_list)
        # self.out_conv = nn.Conv2d(in_channels=15,out_channels=13,kernel_size=1,stride=1,padding=0)

    def forward(self, data, omega):

        # Observation
        OmegaD = data
        Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        OmegaD = torch.mul(OmegaD, omega)
        C  = OmegaD
        L1 = torch.zeros(C.size(), device=torch.device('cuda'))
        L2 = torch.zeros(C.size(), device=torch.device('cuda'))
        L3 = torch.zeros(C.size(), device=torch.device('cuda'))
        E  = torch.zeros(C.size(), device=torch.device('cuda'))
        T  = torch.zeros(C.size(), device=torch.device('cuda'))
        P  = torch.zeros(C.size(), device=torch.device('cuda'))
        Q  = torch.zeros(C.size(), device=torch.device('cuda'))
        # Init L/R
        L = torch.ones((15, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, C, E, T, P, Q, L1, L2, L3] = self.network[i](L, R, C, E, T, P, Q, L1, L2, L3, OmegaD, W, Omega_C)
            layers.append([L, R, C, E, T, P, Q, L1, L2, L3])

        # C_out = self.out_conv(layers[-1][2])

        return C

class RPCA_Net_3(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net_3, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(15)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_2())
        self.network = nn.ModuleList(blocks_list)
        self.out_conv = nn.Conv2d(in_channels=15,out_channels=13,kernel_size=1,stride=1,padding=0)

    def forward(self, data, omega):

        # Observation
        OmegaD = data
        Omega_C = torch.tensor(1.).cuda() -  omega

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        OmegaD = torch.mul(OmegaD, omega)
        C  = OmegaD
        L1 = torch.zeros(C.size(), device=torch.device('cuda'))
        L2 = torch.zeros(C.size(), device=torch.device('cuda'))
        L3 = torch.zeros(C.size(), device=torch.device('cuda'))
        E  = torch.zeros(C.size(), device=torch.device('cuda'))
        T  = torch.zeros(C.size(), device=torch.device('cuda'))
        P  = torch.zeros(C.size(), device=torch.device('cuda'))
        Q  = torch.zeros(C.size(), device=torch.device('cuda'))
        # Init L/R
        L = torch.ones((15, 10, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((10, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, C, E, T, P, Q, L1, L2, L3] = self.network[i](L, R, C, E, T, P, Q, L1, L2, L3, OmegaD, W, Omega_C)
            layers.append([L, R, C, E, T, P, Q, L1, L2, L3])

        C_out = self.out_conv(layers[-1][2])

        return C_out