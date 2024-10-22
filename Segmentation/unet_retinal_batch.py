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
import torchvision.transforms as TF

from retinal_loaders import DRIVEDataLoader
from Unet_architecture import UNet2, UNet



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


size = 128
train_transform = transforms.Compose([
    # transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([
    # transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 5
crop_size = (size,size)
trainset = PATCH_DRIVEDataLoader(train=True, transforms=train_transform,crop_size=crop_size )
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = PATCH_DRIVEDataLoader(train=False, transforms=test_transform, crop_size=crop_size)
test_loader = DataLoader(testset, shuffle=False, num_workers=3)


print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def focal_loss(y_real, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    """
    # Apply sigmoid to logits to get probabilities
    
    
    # Flatten the tensors
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)

    # Compute the binary cross-entropy (BCE) loss
    loss = F.binary_cross_entropy_with_logits(y_pred, y_real, reduction="none")

    # Compute the focal loss factor (1 - pt)^gamma
    pt = torch.where(y_real == 1, y_pred, 1 - y_pred)  # p_t = y_pred for positive class, 1-y_pred for negative class
    focal_weight = (1 - pt) ** gamma

    # Apply alpha weighting for the minority class
    alpha_weight = torch.where(y_real == 1, alpha, 1 - alpha)

    # Final focal loss
    loss = focal_weight * alpha_weight * loss

    return loss.mean()
# def focal_loss(y_real, y_pred,gamma = 2.):
#     y_pred_sig = torch.sigmoid(y_pred)
#     term = (1-y_pred_sig)**gamma * y_real * torch.log(y_pred_sig) + (1-y_real) * torch.log(1-y_pred_sig)
#     return (-term.sum())
# def bce_loss(y_real, y_pred):
#     return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss.item() / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        # model.eval()  # testing mode
        # Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        # clear_output(wait=True)
        # for k in range(6):
        #     plt.subplot(3, 6, k+1)
        #     plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
        #     plt.title('Real')
        #     plt.axis('off')

        #     plt.subplot(3, 6, k+7)
        #     plt.imshow(Y_hat[k, 0], cmap='gray')
        #     plt.title('Output')
        #     plt.axis('off')
        #     plt.subplot(3, 6, k+13)
        #     plt.imshow(Y_test[k, 0].detach().cpu(), cmap='gray')
        #     plt.title('Ground Truth')
        #     plt.axis('off')
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        # plt.show()

def predict(model, dataLoader):
    model.eval()  # testing mode
    Y_pred = [torch.sigmoid(model(X_batch.to(device))).detach().cpu() for X_batch, _ in dataLoader]
    return np.array(Y_pred)

def predict2(model, dataLoader, loss_fn = bce_loss):
    avg_loss = 0.
    for X_batch, Y_batch in dataLoader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            torch.sigmoid(model(X_batch.to(device))).detach().cpu()

            # forward
            Y_pred = model(X_batch.to(device))
            loss = loss_fn(Y_batch, Y_pred).detach().cpu()  # forward-pass
             # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
    return avg_loss


    
    
model = UNet2().to(device)
summary(model, (3, 128, 128))

torch.cuda.empty_cache()
train(model, optim.Adam(model.parameters(), 0.00001, weight_decay=0.001), focal_loss, 100, train_loader, test_loader)



accuracy = 0.

predictions = predict2(model, test_loader)
print(f"Predictions: {predictions}")
print(predictions.sum() / len(train_loader))


# We save our compiled model. 
# This way we don't have to have the exact model defined, when we load it later
model_scripted = torch.jit.script(model)
model_scripted.save("model.pt")