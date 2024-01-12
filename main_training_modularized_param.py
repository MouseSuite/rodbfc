import glob
import nibabel as nib
import numpy as np
import datetime
import torch
from torch.nn import MSELoss
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader, CacheDataset, pad_list_data_collate
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, RandAffined, RandBiasFieldd
from monai.utils import set_determinism
import random
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the model with different parameter combinations.")
    parser.add_argument('--channels', type=str, default="16,64,64,128,256", help='List of channels for the UNet model separated by commas.')
    parser.add_argument('--augmentation', type=bool, default=True, help='Whether to use data augmentation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--num_epochs', type=int, default=20002, help='Number of epochs for training.')
    parser.add_argument('--save_interval', type=int, default=500, help='Interval for saving model and loss data.')
    
    args = parser.parse_args()
    
    # Convert the channels argument from string to a list of integers
    args.channels = [int(channel) for channel in args.channels.split(',')]
    
    return args


def main():
    args = parse_arguments()
    
    # Call the main_training_modularized_param function with command-line arguments
    main_training_modularized_param(args.channels, args.augmentation, args.lr, args.num_epochs, args.save_interval)



def main_training_modularized_param(channels, augmentation, lr, num_epochs, save_interval):

    folder_name = f"channels_{channels}_aug_{augmentation}_lr_{lr}"
    directory_path = os.path.join('/project/ajoshi_27/code_farm/rodbfc/models', folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Set determinism
    set_determinism(seed=0)
    random.seed(11)

    # File paths
    uncorr_files = glob.glob('/project/ajoshi_27/rodent_bfc_data4ML/data4ML2/*uncorr.nii.gz')
    #calculate_and_save_ratio(uncorr_files)

    image_files = uncorr_files
    bias_files = []
    for f in uncorr_files:
        sub_name = f[:-14]
        bias_files.append(sub_name + '_bias.nii.gz')


    # Define transformations based on augmentation
    train_transforms = get_train_transforms(augmentation)
    val_transforms = get_val_transforms()

    # Create datasets and dataloaders
    train_loader, val_loader = create_dataset_and_loader(image_files, bias_files, train_transforms, val_transforms)

    # Train the model
    train_model(train_loader, val_loader, channels, lr, num_epochs, save_interval,directory_path)

def get_train_transforms(augmentation):
    keys = ["image", "bias"]
    transforms_list = [
        LoadImaged(keys, image_only=True),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        Resized(keys, spatial_size=(64, 64, 64), mode='trilinear'),    
    ]

    if augmentation:
        transforms_list.extend([
            RandAffined(
                keys,
                prob=0.5,
                rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
                translate_range=(15, 15, 15),
                scale_range=(0.3, 0.3, 0.3),
                shear_range=(.1,.1,.1,.1,.1,.1),
                padding_mode=("zeros", "reflection"),
            ),
            RandBiasFieldd(keys, prob=0.5, coeff_range=(-1, 1), degree=5),
        ])

    return Compose(transforms_list)

def get_val_transforms():
    keys = ["image", "bias"]
    return Compose([
        LoadImaged(keys, image_only=True),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        Resized(keys, spatial_size=(64, 64, 64), mode='trilinear'),
    ])



# Function to train the model
def train_model(train_loader, val_loader, channels,lr, num_epochs, save_interval,directory_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,#(16, 64, 64, 128, 256),#(2,8,8,16,32),
        strides=(1, 1, 1, 1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#1e-4)
    loss_function = MSELoss()

    num_epochs = num_epochs#20002
    save_interval = save_interval#500
    train_loss_epoch = np.zeros(num_epochs)
    val_loss_epoch = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            inputs, biases = batch['image'].to(device), torch.log(batch['bias']).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, biases)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, biases = batch['image'].to(device), torch.log(batch['bias']).to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, biases)
                total_val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_train_loss / len(train_loader)}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss / len(val_loader)}")

        # Saving model and loss data
        if epoch % save_interval == 0:
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            model_filename = f"bias_field_correction_model_{formatted_datetime}_epoch_{epoch}.pth"
            loss_filename = f"bias_field_correction_loss_{formatted_datetime}_epoch_{epoch}.npz"
            model_filename = os.path.join(directory_path, model_filename)
            loss_filename = os.path.join(directory_path, loss_filename)
            torch.save(model.state_dict(), model_filename)
            np.savez(loss_filename, val_loss_epoch=val_loss_epoch, train_loss_epoch=train_loss_epoch)


# Function to calculate ratio and save the NIfTI image
def calculate_and_save_ratio(uncorr_files):
    for f in uncorr_files:
        sub_name = f[:-14]
        corr_file = sub_name + '_corr.nii.gz'
        uncorr_file = sub_name + '_uncorr.nii.gz'
        output_file = sub_name + '_bias.nii.gz'

        image1 = nib.load(corr_file).get_fdata()
        image2 = nib.load(uncorr_file).get_fdata().squeeze()

        ratio = np.divide(image2, image1 + 0.1, out=np.zeros_like(image2))
        ratio[ratio > 10] = 1

        ratio_image = nib.Nifti1Image(ratio, affine=nib.load(uncorr_file).affine)
        nib.save(ratio_image, output_file)
        print(f"Ratio image saved as {output_file}")


# Function to create datasets and dataloaders
def create_dataset_and_loader(image_files, bias_files, train_transforms, val_transforms):
    data_dicts = [{"image": image, "bias": bias} for image, bias in zip(image_files, bias_files)]
    train_ds = CacheDataset(data=data_dicts[:round(0.8 * len(data_dicts))], transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=data_dicts[round(0.8 * len(data_dicts)):], transform=val_transforms, cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=4, num_workers=10, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=10, collate_fn=pad_list_data_collate)

    return train_loader, val_loader



if __name__ == "__main__":
    main()
