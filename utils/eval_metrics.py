import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity
from geomloss import SamplesLoss
import lpips


class ManifoldDistance(nn.Module):

    def __init__(self, lambda_color=0.01):
        super(ManifoldDistance, self).__init__()
        self.lambda_color = lambda_color
        # Set scaling=1.2 instead of None
        self.emd_loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=0.01,
            scaling=1.2,
            cost=self.cost_function,
            reach=None,  # Use None so geomloss handles truncation
            debias=False,  # Disable debias for better stability
        )


    def cost_function(self, x, y):
        # Split coordinates and colors
        coords_x, color_x = x[:,:,:3], x[:,:,3:]
        coords_y, color_y = y[:,:,:3], y[:,:,3:]

        # Compute normalization factor
        max_norm = torch.norm(torch.cat([coords_x, coords_y], dim=1), dim=-1).max()
        norm = 1.0 / (max_norm + 1e-6)

        # Normalize coordinates
        coords_x_normalized = coords_x * norm
        coords_y_normalized = coords_y * norm

        # Compute coordinate distance matrix (B, N, M)
        coord_dist = torch.sum(
            (coords_x_normalized.unsqueeze(2) - coords_y_normalized.unsqueeze(1)) ** 2,
            dim=-1
        )

        # Compute color distance matrix; clamp colors to [0, 1]
        color_x = torch.clamp(color_x, 0, 1)
        color_y = torch.clamp(color_y, 0, 1)
        color_dist = torch.sum(
            (color_x.unsqueeze(2) - color_y.unsqueeze(1)) ** 2,
            dim=-1
        )

        # Combine distances and ensure non-negative
        cost_matrix = coord_dist + self.lambda_color * color_dist
        return torch.clamp(cost_matrix, min=0.0)  # Ensure cost is non-negative


    def forward(self, x, y):
        weights_a = torch.ones(x.shape[0], x.shape[1], device=x.device) / x.shape[1]
        weights_b = torch.ones(y.shape[0], y.shape[1], device=y.device) / y.shape[1]
        try:
            distance = self.emd_loss(weights_a, x, weights_b, y).mean()
        except Exception as e:
            print(f"EMD computation error: {e}")
            print(x.max(), x.min(), y.max(), y.min())
            distance = torch.tensor(float('nan'), device=x.device)
        return distance * 10  # Scale for better readability


def param_dist(x, y, normalize=False):
    """
    Use L1 distance to measure parameter distance using NumPy
    Input x, y should be of shape (N, 59) or (B, N, 59)
    """
    assert x.shape[-1] == 59 and y.shape[-1] == 59, "Input should be of shape (N, 59) or (B, N, 59)"
    
    # Convert PyTorch tensors to NumPy if needed
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # Handle NaN and Inf values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    if normalize:
        # Calculate statistics
        mean_x, max_x, min_x = np.mean(x, axis=0), np.max(x, axis=0), np.min(x, axis=0)
        mean_y, max_y, min_y = np.mean(y, axis=0), np.max(y, axis=0), np.min(y, axis=0)
        mean = (mean_x + mean_y) / 2
        max_val = (max_x + max_y) / 2
        min_val = (min_x + min_y) / 2
        
        # Prevent division by zero with a larger epsilon
        epsilon = 1e-6
        denominator = np.maximum(max_val - min_val, epsilon)
        
        # Apply normalization
        x = (x - mean) / denominator
        y = (y - mean) / denominator
    
    # Reshape to 2D arrays
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    # Calculate L1 distance
    divisor = max(x.shape[1] * x.shape[0], 1)  # Ensure non-zero divisor
    l1_dist = np.abs(x - y).sum(axis=1, keepdims=True) / divisor
    
    # Final check for inf and return
    return np.nan_to_num(l1_dist, nan=0.0, posinf=1.0, neginf=0.0)


def mse(x, y):
    # Handle NaN and Inf values
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate MSE
        squared_diff = ((x - y) ** 2)
        
        # Reshape and calculate mean
        reshaped = squared_diff.view(x.shape[0], -1)
        
        # Handle empty tensor or all-zero case
        if reshaped.numel() == 0 or torch.all(reshaped == 0):
            return torch.zeros((x.shape[0], 1), device=x.device)
            
        return reshaped.mean(1, keepdim=True)
    else:
        # NumPy implementation 
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        squared_diff = ((x - y) ** 2)
        reshaped = squared_diff.reshape(x.shape[0], -1)
        
        if reshaped.size == 0 or np.all(reshaped == 0):
            return np.zeros((x.shape[0], 1))
            
        return reshaped.mean(1, keepdims=True)


def psnr(img1, img2, max_val=1.0):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    mse = torch.clamp(mse, min=epsilon)  # Prevent zero MSE
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2):
    img1_gray = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
    img2_gray = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    ssim_value = structural_similarity(img1_gray, img2_gray, 
                                       data_range=img2_gray.max() - img2_gray.min())
    return ssim_value


def lpips_loss(img1, img2, net='vgg'):
    """
    Compute LPIPS loss between two batches of images.
    
    Args:
        img1: Tensor of shape (B, C, H, W) with values in [0, 1]
        img2: Tensor of shape (B, C, H, W) with values in [0, 1]
        net: Backbone network for LPIPS ('vgg', 'alex', 'squeeze')
        
    Returns:
        Tensor of shape (B, 1) with LPIPS loss for each image pair
    """
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net).to(img1.device)
    
    # Ensure input images are in the correct range [-1, 1]
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    
    # Compute LPIPS loss
    with torch.no_grad():  # Disable gradient computation for evaluation
        lpips_values = lpips_model(img1, img2)
    
    return lpips_values
