# DeblurGAN-WebFlask
A user-friendly flask web application where users are able to deblur a motion blurred images using.
# Pseudocode for Training and Testing DeblurGAN

# Prepare the Dataset

    #Download Dataset

    # Download the GOPRO dataset from Seungjun Nah.
      #  https://seungjunnah.github.io/Datasets/gopro

# Organize Dataset
 
#         DeblurGAN/
# ├── blurred_sharp/
#     ├── blurred/
#     │   ├── train/
#     │   └── test/
#     └── sharp/
#         ├── train/
#         └── test/
'''
Place 2103 blurred images in blurred_sharp/blurred/train/.

Place 2103 sharp images in blurred_sharp/sharp/train/.

 Combine Images

    Run the combine_A_and_B.py script to create a combined dataset
'
#/home/janadal/DeblurGAN2/datasets/combine_A_and_B.py --fold_A "/home/janadal/DeblurGAN2/blurred_sharp/blurred/train" --fold_B "/home/janadal/DeblurGAN2/blurred_sharp/sharp/train" --fold_AB "/home/janadal/DeblurGAN2/blurred_sharp/combined"

'
 Setup the Environment

    Create and Activate Conda Environment
    Create a Conda virtual environment.
'''
    # For visualizing training 
# activate conda
    #conda activate deblurgan_env
'
Install Required Libraries

Install necessary Python libraries using pip or conda.
Run Visdom Server

Start the Visdom server for monitoring training
'
# To enable Visdom Server
    #python -m visdom.server
'
Train the Model
    Run Training Script
        Execute the training script
'
# python train.py --dataroot "DeblurGAN/blurred_sharp/combine" --learn_residual --resize_or_crop crop --fineSize 256

#Training will proceed for the specified number of epochs a total of (300 epochs)
'
Test the Model
    Prepare Test Images
        Create a directory for test images
'
#DeblurGAN/
#└── Test/

'''
Place the blurry images to be deblurred in the Test/ directory.
Run Testing Script
Execute the testing script.
'''
# python test.py --which_epoch 95 --dataroot "/home/janadal/DeblurGAN3/Test6" --model test --dataset_mode single --learn_residual

# General packages
numpy==1.24.2
Pillow==9.4.0
scikit-image==0.21.0
scikit-learn==1.2.2
torch==2.0.1+cu118
torchvision==0.15.2+cu118
flask==2.3.4
werkzeug==2.3.4
visdom==1.4.8

# Additional packages
tqdm==4.65.0

'-----------------------------------------------------------------------------------------------------'

'   
    Run the command below
'
# conda activate deblurgan_env
# cd app
# cd DEBG
# python -m visdom.server

' 
    Then Open browser and new terminal again
'
# conda activate deblurgan_env
# python train.py --dataroot "/home/janadal/DeblurGAN3/app/DEBG/blurred_sharp/combined" --learn_residual --resize_or_crop crop --fineSize 512 --niter 200 --niter_decay 200
' To continue the training'
# python train.py --dataroot "/home/janadal/DeblurGAN3/app/DEBG/blurred_sharp/combined" --learn_residual --resize_or_crop crop --fineSize 512 --niter 200 --niter_decay 200 --checkpoints_dir /home/janadal/DeblurGAN3/app/DEBG/checkpoints --name experiment_name --continue_train --epoch_count 186

'Testing'

# python test.py --which_epoch 30 --dataroot "/home/janadal/DeblurGAN2/Test7" --model test --dataset_mode single --learn_residual --resize_or_crop crop --fineSize 512

'Test the whole image'

# python test.py --which_epoch latest --dataroot "/home/janadal/DeblurGAN3/Test6" --model test --dataset_mode single --learn_residual --resize_or_crop resize --loadSizeX 1280 --loadSizeY 720
#python test_select_region.py --which_epoch latest --dataroot "/home/janadal/DeblurGAN3/Test8" --model test --dataset_mode single --learn_residual --resize_or_crop resize --loadSizeX 256 --loadSizeY 256
'change --loadSizeX 1280 --loadSizeY 720 depends on the image resolution'
