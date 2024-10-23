import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output

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

size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 25
trainset = DRIVEDataLoader(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = DRIVEDataLoader(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))

plt.rcParams['figure.figsize'] = [18, 6]

images, labels = next(iter(train_loader))

for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

    plt.subplot(2, 6, i+7)
    plt.imshow(labels[i].squeeze())
plt.show()