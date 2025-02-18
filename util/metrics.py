import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def SSIM(img1, img2):
	# Convert inputs to PyTorch tensors if they're numpy arrays
	if isinstance(img1, np.ndarray):
		img1 = torch.from_numpy(img1).float() / 255.0  # Normalize to [0, 1]
	if isinstance(img2, np.ndarray):
		img2 = torch.from_numpy(img2).float() / 255.0  # Normalize to [0, 1]

	# Ensure the images are in the correct shape (C, H, W)
	if len(img1.shape) == 3:  # Shape: (H, W, C)
		# Permute the dimensions from (H, W, C) to (C, H, W)
		img1 = img1.permute(2, 0, 1)
		img2 = img2.permute(2, 0, 1)
	
	# Add batch dimension if not present
	if len(img1.shape) == 3:
		img1 = img1.unsqueeze(0)
		img2 = img2.unsqueeze(0)

	_, channel, _, _ = img1.shape

	window_size = 11
	window = create_window(window_size, channel)

	# Move window to the same device as input images
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
	denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

	# Avoid division by zero
	ssim_map = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
	
	# Return the mean SSIM
	return ssim_map.mean().item()

def PSNR(img1, img2):
	# Ensure the images are in the range [0, 255]
	img1 = img1 / 255.0 if img1.max() > 1 else img1
	img2 = img2 / 255.0 if img2.max() > 1 else img2

	mse = np.mean((img1 - img2) ** 2)  # Calculate MSE directly
	if mse == 0:
		return 100  # Return 100 if images are identical
	PIXEL_MAX = 1  # Set to 1 for normalized images
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Main function to calculate metrics
def calculate_metrics(img1, img2):
	ssim_value = SSIM(img1, img2)
	psnr_value = PSNR(img1, img2)
	
	return ssim_value.item(), psnr_value.item()
