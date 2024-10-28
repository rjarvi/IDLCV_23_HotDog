#!/zhome/44/9/212447/venv_1/bin/python3
import numpy as np
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as TF
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output
from retinal_loaders import PATCH_DRIVEDataLoader
from model import UNet2, UNet3,UNet2Dilated, EncDec
from training_func import train_with_validation
from loss_functions import dice_loss,focal_loss, bce_loss, dice_coefficient, intersection_over_union, accuracy, sensitivity, specificity, evaluate_model_with_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


rotation = 50
size = 512
transform = TF.Compose([
    TF.RandomHorizontalFlip(p=0.5),
    TF.RandomRotation(rotation),
    TF.Resize((512, 512)),  # Resize to a fixed size
    # transforms.ToTensor() # Dataloader handles this by itself
])
# Create dataset instances for train, validation, and test
crop = (512, 512)

train_dataset = PATCH_DRIVEDataLoader(split="Train",    transforms=transform, crop_size=None)
val_dataset = PATCH_DRIVEDataLoader(split="Validation", transforms=transform, crop_size=None)
test_dataset = PATCH_DRIVEDataLoader(split="Test",      transforms=transform, crop_size=None)

batch_size = 1

#Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")


model = UNet2Dilated(32).to(device)
# summary(model, (3, 512, 512))

train_with_validation(
    device, 
    model, 
    optim.Adam(model.parameters(),
               weight_decay=1e-4),
    dice_loss, 
    250, 
    train_loader, 
    val_loader, 
    test_loader,
    patience=30)

#do evaluate model performace on loss functios

metrics = evaluate_model_with_metric(model, device, test_loader)


avg_dice = metrics["dice"]
print(f'Final Model Performance - Dice Coefficient Metric: {avg_dice:.4f}')
intersect = metrics["iou"]
print(f'Final Model Performance - Intersection Over Union Metric: {intersect:.4f}')
accuracy_ = metrics["accuracy"]
print(f'Final Model Performance - Accuracy Metric: {accuracy_:.4f}')
sensitivity_ = metrics["sensitivity"]
print(f'Final Model Performance - Sensitivity Metric: {sensitivity_:.4f}')
specificity_ = metrics["specificity"]
print(f'Final Model Performance - Specificity Metric: {specificity_:.4f}')



