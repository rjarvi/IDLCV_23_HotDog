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
from model import UNet2
from dataloader import SkinLesionLoader

from loss_functions import evaluate_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_path = "/dtu/datasets1/02516//PH2_Dataset_images"
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a fixed size
    transforms.ToTensor()             # Convert to tensor
])
# Create dataset instances for train, validation, and test
test_dataset = SkinLesionLoader(transform=transform, dataset_path=data_path, split='test')

#Create data loaders
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Instantiate the model
model = UNet2()
# Load the state dict
model.load_state_dict(torch.load('best_model.pth'))

model.to(device)  # Move the model to the GPU if available
# Set to evaluation mode
model.eval()
print("Evaluating Model: ")
metrics = evaluate_model(model, test_loader, device)
for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")  # Format to 4 decimal places