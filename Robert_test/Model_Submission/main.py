#!/zhome/44/9/212447/venv_1/bin/python3
import numpy as np
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
from dataloader import SkinLesionLoader
from model import UNet2, EncDec, EncDec2
from training_func import train_with_validation
from loss_functions import bce_loss, evaluate_model, dice_loss, focal_loss
#from loss_functions import dice_coefficient, intersection_over_union, accuracy, sensitivity, specificity, evaluate_model_with_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_path = "/dtu/datasets1/02516//PH2_Dataset_images"
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a fixed size
    transforms.ToTensor()             # Convert to tensor
])
# Create dataset instances for train, validation, and test
train_dataset = SkinLesionLoader(transform=transform, dataset_path=data_path, split='train')
val_dataset = SkinLesionLoader(transform=transform, dataset_path=data_path, split='val')
test_dataset = SkinLesionLoader(transform=transform, dataset_path=data_path, split='test')

#Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")


#model = UNet2().to(device)
model = EncDec2().to(device)
summary(model, (3, 256, 256))


# Define a weight for the positive class (e.g., 2.0 for imbalance)
pos_weight = torch.tensor([2.0]).to(device)  # Adjust based on your dataset

# Instantiate the loss function
#BCE_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#train_with_validation(device, model, optim.Adam(model.parameters()), bce_loss, 20, train_loader, val_loader, test_loader)
train_with_validation(device, model, optim.Adam(model.parameters()), dice_loss, 20, train_loader, val_loader, test_loader)
#do evaluate model performace on loss functios

#avg_dice = evaluate_model_with_metric(model, device, test_loader, dice_coefficient)
# print(f'Final Model Performance - Dice Coefficient Metric: {avg_dice:.4f}')
# intersect = evaluate_model_with_metric(model, device, test_loader, intersection_over_union)
# print(f'Final Model Performance - Intersection Over Union Metric: {intersect:.4f}')
# accuracy_ = evaluate_model_with_metric(model, device, test_loader, accuracy)
# print(f'Final Model Performance - Accuracy Metric: {accuracy_:.4f}')
# sensitivity_ = evaluate_model_with_metric(model, device, test_loader, sensitivity)
# print(f'Final Model Performance - Sensitivity Metric: {sensitivity_:.4f}')
# specificity_ = evaluate_model_with_metric(model, device, test_loader, specificity)
# print(f'Final Model Performance - Specificity Metric: {specificity_:.4f}')

print("Final Model Performance")
metrics = evaluate_model(model, test_loader, device)
for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")  # Format to 4 decimal places