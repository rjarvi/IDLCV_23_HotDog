#!/zhome/44/9/212447/venv_1/bin/python3
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

from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

data_path = '/dtu/datasets1/02516/DRIVE'
#data_path = './phc_data'
# Adjusted data_path to match your directory structure

class DRIVEDataLoader(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, data_path='/dtu/datasets1/02516/DRIVE'):
        'Initialization'
        self.transform = transform
        if train:
            # Training data
            image_folder = os.path.join(data_path, 'training', 'images')
            mask_folder = os.path.join(data_path, 'training', '1st_manual')
        else:
            # Test data
            image_folder = os.path.join(data_path, 'test', 'images')
            mask_folder = os.path.join(data_path, 'test', 'mask')

        # Get image and mask paths (tif for images, gif for masks)
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.tif')))
        self.label_paths = sorted(glob.glob(os.path.join(mask_folder, '*.gif')))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the image and the mask
        image = Image.open(image_path)
        label = Image.open(label_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# Define the transforms (resizing to 128x128 for example)
size = 128
train_transform = transforms.Compose([
    transforms.Resize((size, size)), 
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((size, size)), 
    transforms.ToTensor()
])

# Set batch size
batch_size = 6

# Create the full training set
full_trainset = DRIVEDataLoader(train=True, transform=train_transform)

# Define the train-validation split ratio (80% train, 20% validation)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

# Split the dataset into training and validation sets
trainset, valset = random_split(full_trainset, [train_size, val_size])

# Create the test set
testset = DRIVEDataLoader(train=False, transform=test_transform)

# Create the DataLoaders for train, validation, and test sets
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

# Print data sizes
print(f'Loaded {len(trainset)} training images')
print(f'Loaded {len(valset)} validation images')
print(f'Loaded {len(testset)} test images')


# Example of loading a batch of images from the training set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def check_data_structure(data_path):
    for root, dirs, files in os.walk(data_path):
        # Print current directory path
        print(f"Directory: {root}")
        
        # Print subdirectories
        for subdir in dirs:
            print(f"  Subdirectory: {subdir}")
        
        # Print files
        for file in files:
            print(f"  File: {file}")

# Call the function to check the structure
check_data_structure(data_path)