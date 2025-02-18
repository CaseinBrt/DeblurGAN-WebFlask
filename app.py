from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import os
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from util import util
from PIL import Image
from util import metrics
import numpy as np
import torch
import json
from functools import lru_cache
from torchvision import transforms
import time


app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Upload and results folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize the options
opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.model = 'test'
opt.dataset_mode = 'single'
opt.learn_residual = True
opt.resize_or_crop = 'resize'
opt.dataroot = app.config['UPLOAD_FOLDER']

model = None
visualizer = None

def get_model():
    global model, visualizer
    if model is None:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        
        # Set the device for the options
        opt.gpu_ids = [3]
        opt.device = device
        
        model = create_model(opt)
        visualizer = Visualizer(opt)
        
        # Move the model to the specified device
        model.netG.to(device)
        
    return model

@lru_cache(maxsize=32)
def process_image(filename):
    model = get_model()
    original_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(original_filename).convert('RGB')
    original_size = img.size
    
    # Resize the image to match the model's expected input size
    model_input_size = (512, 512) 
    img_resized = img.resize(model_input_size, Image.BICUBIC)
    
    # Convert the image to a tensor
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
    
    # Move the input tensor to the same device as the model
    device = next(model.netG.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Process the image through the DeblurGAN model
    with torch.no_grad():
        deblurred_tensor = model.netG(img_tensor)
    
    # Move the result back to CPU for further processing
    deblurred_tensor = deblurred_tensor.cpu()
    
    # Convert the deblurred tensor back to an image
    deblurred_img = util.tensor2im(deblurred_tensor)
    deblurred_img_pil = Image.fromarray(deblurred_img)
    
    # Resize the deblurred image back to its original size
    deblurred_img_pil = deblurred_img_pil.resize(original_size, Image.BICUBIC)
    
    result_filename = os.path.join(app.config['RESULTS_FOLDER'], 'deblurred_' + filename)
    deblurred_img_pil.save(result_filename)
    
    # Calculate PSNR and SSIM
    original_img = np.array(Image.open(original_filename))
    deblurred_img = np.array(deblurred_img_pil)
    psnr = metrics.PSNR(original_img, deblurred_img)
    ssim = metrics.SSIM(original_img, deblurred_img)
    
    return result_filename, psnr, ssim

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Add timestamp to filename
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            
            # Save the original image with timestamp
            original_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_filename)
            
            result_filename, psnr, ssim = process_image(filename)
            
            # This part will check if the deblurring results meet the threshold
            if psnr < 23:
                return jsonify({
                    'warning': True,
                    'original': filename,
                    'result': os.path.basename(result_filename),
                    'psnr': float(psnr),
                    'ssim': float(ssim)
                })
            
            return jsonify({
                'warning': False,
                'original': filename,
                'result': os.path.basename(result_filename),
                'psnr': float(psnr),
                'ssim': float(ssim)
            })
    return render_template('index.html')

@app.route('/deblur_region', methods=['POST'])
def deblur_region():
    file = request.form['file']
    region = json.loads(request.form['region'])
    
    # Load the original image
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], file))
    
    # Crop the selected region
    region_img = img.crop((region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height']))
    
    # Resize the region to match the model's expected input size
    model_input_size = (256, 256)  
    region_img_resized = region_img.resize(model_input_size, Image.BICUBIC)
    
    # Convert the region to a tensor
    region_tensor = transforms.ToTensor()(region_img_resized).unsqueeze(0)
    
    # This will get the model
    model = get_model()
    
    # Move the input tensor to the same device as the model
    device = next(model.netG.parameters()).device
    region_tensor = region_tensor.to(device)
    
    # Process the region through the DeblurGAN model
    with torch.no_grad():
        deblurred_region_tensor = model.netG(region_tensor)
        # Enhance the output
        deblurred_region_tensor = torch.clamp(deblurred_region_tensor * 1.2, -1, 1)
    
    # Same with the entire image deblurring move the result back to CPU for further processing
    deblurred_region_tensor = deblurred_region_tensor.cpu()
    
    # Convert the deblurred region back to an image
    deblurred_region = util.tensor2im(deblurred_region_tensor)
    deblurred_region_img = Image.fromarray(deblurred_region)
    
    # Resize the deblurred region back to its original size
    deblurred_region_img = deblurred_region_img.resize((region['width'], region['height']), Image.BICUBIC)
    
    # Paste the deblurred region back into the original image
    result_img = img.copy()
    result_img.paste(deblurred_region_img, (region['x'], region['y']))
    
    # Save the result with timestamp
    timestamp = int(time.time())
    result_filename = f'deblurred_region_{timestamp}_{file}'
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    result_img.save(result_path)
    
    # Calculate PSNR and SSIM
    original_array = np.array(img)
    result_array = np.array(result_img)
    psnr = metrics.PSNR(original_array, result_array)
    ssim = metrics.SSIM(original_array, result_array)
    
    # Restriction condition on PSNR
    if psnr < 23:
        return jsonify({
            'warning': True,
            'result': url_for('result_file', filename=result_filename),
            'psnr': float(psnr),
            'ssim': float(ssim)
        })
    
    return jsonify({
        'warning': False,
        'result': url_for('result_file', filename=result_filename),
        'psnr': float(psnr),
        'ssim': float(ssim)
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/compare', methods=['POST'])
def compare():
    entire_filename = request.form['entire_filename']
    region_filename = request.form['region_filename']
    
    entire_img = np.array(Image.open(os.path.join(app.config['RESULTS_FOLDER'], entire_filename)))
    region_img = np.array(Image.open(os.path.join(app.config['RESULTS_FOLDER'], region_filename)))
    original_img = np.array(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], entire_filename.replace('deblurred_', ''))))
    
    entire_psnr = metrics.PSNR(original_img, entire_img)
    entire_ssim = metrics.SSIM(original_img, entire_img)
    region_psnr = metrics.PSNR(original_img, region_img)
    region_ssim = metrics.SSIM(original_img, region_img)
    
    better_method = "Entire image deblurring" if entire_psnr > region_psnr else "Region deblurring"
    
    return jsonify({
        'entire_psnr': entire_psnr,
        'entire_ssim': entire_ssim,
        'region_psnr': region_psnr,
        'region_ssim': region_ssim,
        'better_method': better_method
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5005)