import argparse


import numpy as np
import nibabel as nib
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.nn import MSELoss
from monai.data import Dataset, DataLoader, partition_dataset

from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord,LoadImage,AddChannel,ToTensor,EnsureChannelFirstD,EnsureChannelFirstd, Resized, Resize, RandBiasFieldd
from monai.utils import set_determinism
from glob import glob
import random
import matplotlib.pyplot as plt






def main():
    parser = argparse.ArgumentParser(description="Command-line tool for processing input and output filenames.")
    parser.add_argument("-i", "--input", help="Input filename (Uncorrected MRI image filename)", required=True)
    parser.add_argument("-m", "--model", help="Model file (Trained model .pth file)", required=True)
    parser.add_argument("-o", "--output", help="Output filename (Bias corrected MRI image filename)", required=True)
    
    
    args = parser.parse_args()
    
    input_filename = args.input
    output_filename = args.output
    model_filename = args.model
    

    # Define the UNet model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Specify spatial_dims and strides for 3D data
    spatial_dims = 3
    strides = (1, 1, 1, 1)
    
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,  # Adjust based on your data
        out_channels=1, # Adjust based on your data
        channels=(2,8,8,16,32),#(16, 64, 64, 128, 256)/8,
        strides=strides,
    ).to(device)
    
    
    keys=["image"]
    
    test_transforms = Compose([
        LoadImaged(keys,image_only=True),
        EnsureChannelFirstd(keys),
        Resized(
                keys,
                spatial_size=(64, 64, 64),
            ),
       # RandBiasField(prob=1, coeff_range=(0.2,0.3)),
        #ToTensor(),
    ])
    
    model.eval()
    
    # Load the test image (adjust the path to your validation image)
    test_image_path = input_filename
    test_corrected_image_path = output_filename
    model_file=model_filename
    
    model.load_state_dict(torch.load(model_file))
    
    
    test_dict = [{"image":test_image_path}]
    #test_dict = [{"image": image, "bias": bias} for image, bias in zip(image_files, bias_files)]
    # Apply transformations to the validation image
    test_image = test_transforms(test_dict)[0]["image"].to(device)
    
    # Apply the trained model to estimate the bias field
    with torch.no_grad():
        estimated_bias_field = model(test_image[None,])
    
    # Convert the estimated bias field to a Numpy array
    estimated_bias_field = estimated_bias_field.squeeze().cpu().numpy()
    
    # Load the original validation image without resizing (for displaying the corrected image)
    original_test_image = nib.load(test_image_path).get_fdata().squeeze()
    
    orig_shape = original_test_image.shape
    
    
    estimated_bias_field_resized = Resize(spatial_size=orig_shape)(estimated_bias_field[None,])[0]
    
    # Apply the estimated bias field to correct the original image
    corrected_image = original_test_image / np.exp(estimated_bias_field_resized)
    
    input_nifti = nib.load(test_image_path)
    input_dtype = input_nifti.get_data_dtype()
    corrected_image = corrected_image.astype(input_dtype)
    
    
    # Create a new NIfTI image with the result data
    result_nifti = nib.Nifti1Image(corrected_image, input_nifti.affine)
    
    # Save the result as a new NIfTI image
    nib.save(result_nifti, test_corrected_image_path)
    



if __name__ == "__main__":
    main()
