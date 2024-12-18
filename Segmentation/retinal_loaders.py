import os
import glob
import PIL.Image as Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import torchvision.transforms as TF
import torch.nn.functional as F
data_path = '/dtu/datasets1/02516/DRIVE'

class DRIVEDataLoader(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, data_path='/dtu/datasets1/02516/DRIVE'):
        'Initialization'
        self.transform = transform
        self.is_train_loader = train
        if train:
            # Training data
            image_folder = os.path.join(data_path, 'training', 'images')
            vessel_mask_folder = os.path.join(data_path, 'training', '1st_manual')  # Vessel masks for training
            fov_mask_folder = os.path.join(data_path, 'training', 'mask')  # Field of view masks for training
        else:
            # Test data
            image_folder = os.path.join(data_path, 'test', 'images')
            fov_mask_folder = os.path.join(data_path, 'test', 'mask')  # Only FOV masks are available for test set
            vessel_mask_folder = None  # No vessel mask for test set

        # Get image paths
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.tif')))

        # Get vessel mask paths if available (for training)
        if vessel_mask_folder:
            self.vessel_mask_paths = sorted(glob.glob(os.path.join(vessel_mask_folder, '*.gif')))
        else:
            self.vessel_mask_paths = [None] * len(self.image_paths)  # No vessel mask for test set

        # Get FOV mask paths
        self.fov_mask_paths = sorted(glob.glob(os.path.join(fov_mask_folder, '*.gif')))
        
        # Check if we have found the files
        if len(self.image_paths) == 0 or len(self.fov_mask_paths) == 0:
            raise ValueError(f"No images or masks found in the provided path: {data_path}")

        if train and len(self.image_paths) != len(self.vessel_mask_paths):
            raise ValueError("Mismatch between the number of images and vessel masks in the training set.")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        vessel_mask_path = self.vessel_mask_paths[idx]
        fov_mask_path = self.fov_mask_paths[idx]

        # Load the image
        image = Image.open(image_path)

        # Load the vessel mask (if available, i.e., for training)
        vessel_mask = Image.open(vessel_mask_path) if vessel_mask_path else None

        # Load the FOV mask
        fov_mask = Image.open(fov_mask_path)

        # Apply transforms to the image and masks
        if self.transform:
            image = self.transform(image)
            fov_mask = self.transform(fov_mask)
            if vessel_mask:
                vessel_mask = self.transform(vessel_mask)

        # Apply FOV mask to both image and vessel mask if available
        if self.is_train_loader:
            image = image * fov_mask
            vessel_mask = vessel_mask * fov_mask
            return image, vessel_mask  # Return both image and vessel mask for training

        # For test set, return only the FOV-masked image and FOV mask
        return image, fov_mask
    
    
    
    
class PATCH_DRIVEDataLoader(torch.utils.data.Dataset):
    def __init__(self, train=True, transforms=None,crop_size=None, data_path='/dtu/datasets1/02516/DRIVE', window_size=(128,128)):
        'Initialization'
        self.transforms = transforms
        self.is_train_loader = train
        
        self.crop_size = crop_size
        
        
        
        if train:
            # Training data
            image_folder = os.path.join(data_path, 'training', 'images')
            vessel_mask_folder = os.path.join(data_path, 'training', '1st_manual')  # Vessel masks for training
            fov_mask_folder = os.path.join(data_path, 'training', 'mask')  # Field of view masks for training
        else:
            # Test data
            image_folder = os.path.join(data_path, 'test', 'images')
            fov_mask_folder = os.path.join(data_path, 'test', 'mask')  # Only FOV masks are available for test set
            vessel_mask_folder = None  # No vessel mask for test set

        # Get image paths
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.tif')))

        # Get vessel mask paths if available (for training)
        if vessel_mask_folder:
            self.vessel_mask_paths = sorted(glob.glob(os.path.join(vessel_mask_folder, '*.gif')))
        else:
            self.vessel_mask_paths = [None] * len(self.image_paths)  # No vessel mask for test set

        # Get FOV mask paths
        self.fov_mask_paths = sorted(glob.glob(os.path.join(fov_mask_folder, '*.gif')))
        
        # Check if we have found the files
        if len(self.image_paths) == 0 or len(self.fov_mask_paths) == 0:
            raise ValueError(f"No images or masks found in the provided path: {data_path}")

        if train and len(self.image_paths) != len(self.vessel_mask_paths):
            raise ValueError("Mismatch between the number of images and vessel masks in the training set.")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def transform(self, image, vessel_mask, fov_mask):
        
        out_image = image
        out_vessel_mask = vessel_mask
        out_fov_mask = fov_mask
        
        
        
        if not self.is_train_loader:
            resize_params = np.array(out_image).shape
            resize_x = resize_params[0] / self.crop_size[0]
            resize_y = resize_params[1] / self.crop_size[1]
            resize_x = (int)(np.floor(resize_x) * self.crop_size[0])
            resize_y = (int)(np.floor(resize_y) * self.crop_size[1])
            
            out_image = TF.functional.resize(out_image, size=[resize_x, resize_y])
            out_fov_mask = TF.functional.resize(out_fov_mask, size=[resize_x, resize_y])
        
        
        if self.is_train_loader:
            if self.crop_size != None:
                i, j, h, w = TF.RandomCrop.get_params(
                    image, output_size=self.crop_size)
                out_image =         TF.functional.crop(out_image,      i,j,h,w)
                if vessel_mask != None:
                    out_vessel_mask =   TF.functional.crop(out_vessel_mask,i,j,h,w)
                out_fov_mask =      TF.functional.crop(out_fov_mask,   i,j,h,w)
        
        if vessel_mask != None:
            vessel_mask = self.transforms(out_vessel_mask)
        
        return self.transforms(out_image), vessel_mask, self.transforms(out_fov_mask)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        vessel_mask_path = self.vessel_mask_paths[idx]
        fov_mask_path = self.fov_mask_paths[idx]

        # Load the image
        image = Image.open(image_path)

        # Load the vessel mask (if available, i.e., for training)
        vessel_mask = Image.open(vessel_mask_path) if vessel_mask_path else None

        # Load the FOV mask
        fov_mask = Image.open(fov_mask_path)

        image, vessel_mask, fov_mask = self.transform(image, vessel_mask, fov_mask)
        # Apply transforms to the image and masks
        # if self.transform:
        #     image = self.transform(image)
        #     fov_mask = self.transform(fov_mask)
        #     if vessel_mask:
        #         vessel_mask = self.transform(vessel_mask)

        # Apply FOV mask to both image and vessel mask if available
        if self.is_train_loader:
            image = image * fov_mask
            vessel_mask = vessel_mask * fov_mask
            return image, vessel_mask  # Return both image and vessel mask for training

        # For test set, return only the FOV-masked image and FOV mask
        return image, fov_mask