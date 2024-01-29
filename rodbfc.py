#!/usr/bin/env python3

import argparse


import numpy as np
import nibabel as nib
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.nn import MSELoss
from monai.data import Dataset, DataLoader, partition_dataset

from monai.transforms import Compose, ScaleIntensityd, LoadImaged, ToTensord,LoadImage,ToTensor,EnsureChannelFirstD,EnsureChannelFirstd, Resized, Resize, RandBiasFieldd
from monai.utils import set_determinism
from glob import glob
import random
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre






def main():
    parser = argparse.ArgumentParser(description="Command-line tool for processing input and output filenames.")
    parser.add_argument("-i", "--input", help="Input filename (Uncorrected MRI image filename)", required=True)
    parser.add_argument("-m", "--model", help="Model file (Trained model .pth file)", required=True)
    parser.add_argument("-o", "--output", help="Output filename (Bias corrected MRI image filename)", required=True)
    parser.add_argument("-b", "--bias", help="Bias field filename", required=False)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (default: cuda if available, else cpu)")    
    args = parser.parse_args()
    
    input_filename = args.input
    output_filename = args.output
    model_filename = args.model
    bias_filename = args.bias
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = "cpu"
    
    # Use the selected device for computation
    device = torch.device(args.device)

    # Define the UNet model and optimizer
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Specify spatial_dims and strides for 3D data
    spatial_dims = 3
    strides = (1, 1, 1, 1)
    
    last_subdirectory = model_filename.split('/')[-2]
    channels = tuple(map(int, last_subdirectory.split('_')[1][1:-1].split(',')))

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,  # Adjust based on your data
        out_channels=1, # Adjust based on your data
        channels=channels,#(16, 64, 64, 128, 256),#(2,8,8,16,32),#(16, 64, 64, 128, 256),
        strides=strides,
    ).to(device)

    
    keys=["image"]
    
    test_transforms = Compose([
        LoadImaged(keys,image_only=True),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0), 
        # the Unet has instance normalization, so this scaling won't make any difference. 
        # But we keep it for now in case we choose to change the network.    
        Resized(
                keys,
                spatial_size=(64, 64, 64),
                mode='trilinear',
            ),
       # RandBiasField(prob=1, coeff_range=(0.2,0.3)),
        #ToTensor(),
    ])
    
    model.eval()
    
    # Load the test image (adjust the path to your validation image)
    test_image_path = input_filename
    test_corrected_image_path = output_filename
    model_file=model_filename
    
    model.load_state_dict(torch.load(model_file,map_location=torch.device(args.device)))
    
    
    test_dict = [{"image":test_image_path}]
    #test_dict = [{"image": image, "bias": bias} for image, bias in zip(image_files, bias_files)]
    # Apply transformations to the validation image
    test_image = test_transforms(test_dict)[0]["image"].to(device)

    # Apply the trained model to estimate the bias field
    with torch.no_grad():
        print(test_image.max(),test_image.min())
        estimated_bias_field = model(test_image[None,])
    
    # Convert the estimated bias field to a Numpy array
    estimated_bias_field = estimated_bias_field.squeeze().cpu().numpy()
    
    # Load the original validation image without resizing (for displaying the corrected image)
    original_test_image = nib.load(test_image_path).get_fdata().squeeze()
    
    orig_shape = original_test_image.shape
    
    
    estimated_bias_field_resized = Resize(mode='trilinear', spatial_size=orig_shape)(estimated_bias_field[None,])[0]

    estimated_bias_field_resized = np.exp(estimated_bias_field_resized)

    # Smooth the bias field using Legendre polynomials
    order = 32

    # Generate Legendre basis

    # Create Legendre polynomials up to the specified order
    x = torch.linspace(-1, 1, orig_shape[0])
    y = torch.linspace(-1, 1, orig_shape[1])
    z = torch.linspace(-1, 1, orig_shape[2])

    
    legendre_polynomials_x = []
    legendre_polynomials_y = []
    legendre_polynomials_z = []

    for i in range(order + 1):
        basis = Legendre.basis(i)(x.numpy()).astype(np.float32)
        legendre_polynomials_x.append(torch.tensor(basis))
        basis = Legendre.basis(i)(y.numpy()).astype(np.float32)
        legendre_polynomials_y.append(torch.tensor(basis))
        basis = Legendre.basis(i)(z.numpy()).astype(np.float32)
        legendre_polynomials_z.append(torch.tensor(basis))



    for i in range(order + 1):
      legendre_polynomials_x[i] /= np.linalg.norm(legendre_polynomials_x[i])
      legendre_polynomials_y[i] /= np.linalg.norm(legendre_polynomials_y[i])
      legendre_polynomials_z[i] /= np.linalg.norm(legendre_polynomials_z[i])




    estimated_bias_field_resized = torch.tensor(estimated_bias_field_resized, dtype=torch.float32)

    # Project the bias field onto Legendre polynomials
    projected_bias = torch.zeros_like(estimated_bias_field_resized)


    for i in range(order + 1):

        coeff = torch.sum(estimated_bias_field_resized[:,:,:]*legendre_polynomials_x[i][:,None,None],axis=0)
        out = coeff[None,] * legendre_polynomials_x[i][:,None,None]
        projected_bias += out

        coeff = torch.sum(estimated_bias_field_resized*legendre_polynomials_y[i][None,:,None],axis=1)
        out = coeff[:,None,] * legendre_polynomials_y[i][None,:,None]
        projected_bias += out
        
        coeff = torch.sum(estimated_bias_field_resized*legendre_polynomials_z[i][None,None,:],axis=2)
        out = coeff[:,:,None] * legendre_polynomials_z[i][None,None,:]
        
        projected_bias += out
    
    # divide by 3 to take into account the fact that projection on Legendre basis in 3 axis
    projected_bias /= 3.0 
    
    # Apply the estimated bias field to correct the original image
    #projected_bias = estimated_bias_field_resized
    corrected_image = torch.tensor(original_test_image) / projected_bias
    
    input_nifti = nib.load(test_image_path)
    input_dtype = input_nifti.get_data_dtype()
    corrected_image = corrected_image.numpy().astype(np.single)
    
    
    # Create a new NIfTI image with the result data
    result_nifti = nib.Nifti1Image(corrected_image, input_nifti.affine)
    
    # Save the result as a new NIfTI image
    nib.save(result_nifti, test_corrected_image_path)

    # Save the bias field
    if bias_filename is not None:
        bias_nifti = nib.Nifti1Image(projected_bias.numpy().astype(np.single), input_nifti.affine)
        nib.save(bias_nifti, bias_filename)





if __name__ == "__main__":
    main()
