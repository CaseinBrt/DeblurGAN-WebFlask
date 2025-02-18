import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR, SSIM 
import torch
import numpy as np
from PIL import Image

import torch
torch.cuda.empty_cache()


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# This part will create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# Initialize variables to store average PSNR and SSIM
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

# Loop through the dataset and test the model
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    counter = i + 1  # Increment counter
    model.set_input(data)
    model.test()
    
    visuals = model.get_current_visuals()
    
    # Convert visuals to numpy arrays for metric calculations
    fake_B_np = np.array(visuals['fake_B'], dtype=np.float32)
    real_B_np = np.array(visuals['real_A'], dtype=np.float32)
    
    # Convert images to tensors for the custom SSIM calculation
    fake_B_tensor = torch.from_numpy(fake_B_np).unsqueeze(0).permute(0, 3, 1, 2)  # Add batch and channel dimensions
    real_B_tensor = torch.from_numpy(real_B_np).unsqueeze(0).permute(0, 3, 1, 2)
    
    # Calculate PSNR and SSIM
    psnr_value = PSNR(fake_B_np, real_B_np)
    ssim_value = SSIM(fake_B_tensor, real_B_tensor).item()  # Use the correct SSIM function
    
    # Accumulate PSNR and SSIM
    avgPSNR += psnr_value
    avgSSIM += ssim_value
    
    # Print individual PSNR and SSIM for the image
    print(f'Image {i+1} PSNR: {psnr_value}, SSIM: {ssim_value}')
    
    img_path = model.get_image_paths()
    print('Processing image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

# Compute and display average PSNR and SSIM
avgPSNR /= counter
avgSSIM /= counter
print(f'Average PSNR: {avgPSNR}, Average SSIM: {avgSSIM}')

webpage.save()