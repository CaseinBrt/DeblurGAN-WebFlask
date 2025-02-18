from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # Convert tensor to numpy array if it's a torch.Tensor
    if isinstance(image_tensor, torch.Tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor
    
    # If single channel, replicate to 3 channels
    if image_numpy.shape[0] == 1:  # single channel
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    
    # Transpose and convert to specified image type
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # Clip values to [0, 1] range
    image_numpy = np.clip(image_numpy, 0, 1)
    
    # Scale to [0, 255] range
    image_numpy = (image_numpy * 255.0).astype(imtype)
    
    return image_numpy

def save_image(image_numpy, image_path):
    # Check if the image has 3 channels
    if image_numpy.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {image_numpy.shape}")
    
    try:
        # Try to open the original image to get its size
        original_image = Image.open(image_path.replace('deblurred_', ''))
        original_size = original_image.size
    except FileNotFoundError:
        # If original image not found, use the shape of image_numpy
        print(f"Original image not found: {image_path.replace('deblurred_', '')}")
        original_size = image_numpy.shape[:2][::-1]
    
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    # Resize the image to original size
    image_pil = image_pil.resize(original_size, Image.BICUBIC)
    # Save the image
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
	"""Print methods and doc strings.
	Takes module, class, list, dictionary, or string."""
	methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
	processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
	print( "\n".join(["%s %s" %
					 (method.ljust(spacing),
					  processFunc(str(getattr(object, method).__doc__)))
					 for method in methodList]) )

def varname(p):
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

def print_numpy(x, val=True, shp=False):
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
