import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SkinLesionLoader(Dataset):
    def __init__(self, transform, dataset_path, split='train'):
        'Initialization'
        self.transform = transform
        
        # Collect all image paths and label paths
        self.image_paths = []
        self.label_paths = []

        # Loop through all IMG### folders
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.isdir(folder_path):
                # Add dermoscopic image path
                dermoscopic_image_path = os.path.join(folder_path, f'{folder_name}_Dermoscopic_Image', f'{folder_name}.bmp')
                if os.path.exists(dermoscopic_image_path):
                    self.image_paths.append(dermoscopic_image_path)
                
                # Add lesion image path
                lesion_image_path = os.path.join(folder_path, f'{folder_name}_lesion', f'{folder_name}_lesion.bmp')
                if os.path.exists(lesion_image_path):
                    self.label_paths.append(lesion_image_path)

        # Ensure both lists have the same length
        assert len(self.image_paths) == len(self.label_paths), "Image and label counts do not match."

        # Randomly shuffle the dataset
        combined = list(zip(self.image_paths, self.label_paths))
        random.shuffle(combined)
        self.image_paths, self.label_paths = zip(*combined)

        # Split indices for 70-15-15
        total_len = len(self.image_paths)
        train_split = int(total_len * 0.7)
        val_split = int(total_len * 0.85)  # 70% for training, next 15% for validation

        if split == 'train':
            self.image_paths = self.image_paths[:train_split]
            self.label_paths = self.label_paths[:train_split]
        elif split == 'val':
            self.image_paths = self.image_paths[train_split:val_split]
            self.label_paths = self.label_paths[train_split:val_split]
        elif split == 'test':
            self.image_paths = self.image_paths[val_split:]
            self.label_paths = self.label_paths[val_split:]
        else:
            raise ValueError("Invalid split name. Use 'train', 'val', or 'test'.")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
        label = Image.open(label_path).convert("L")    # Ensure label is in grayscale
        
        X = self.transform(image)
        Y = self.transform(label)
        
        return X, Y


# def display_image_pairs(dataset, n_samples=5):
#     sampled_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
#     plt.figure(figsize=(15, 5))
#     for i, idx in enumerate(sampled_indices):
#         # Get image and label
#         image_path = dataset.image_paths[idx]
#         label_path = dataset.label_paths[idx]
        
#         # Open images
#         image = Image.open(image_path).convert("RGB")
#         label = Image.open(label_path).convert("L")  # Convert label to grayscale
        
#         # Display images
#         plt.subplot(2, n_samples, i + 1)
#         plt.imshow(image)
#         plt.title("Dermoscopic Image")
#         plt.axis("off")
        
#         plt.subplot(2, n_samples, i + 1 + n_samples)
#         plt.imshow(label, cmap='gray')
#         plt.title("Lesion Label")
#         plt.axis("off")
    
#     plt.tight_layout()
#     plt.show()

# Display random pairs from the training dataset
# display_image_pairs(train_dataset, n_samples=5)
# display_image_pairs(val_dataset, n_samples=5)
# display_image_pairs(test_dataset, n_samples=5)
