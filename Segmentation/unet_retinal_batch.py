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

from retinal_loaders import DRIVEDataLoader, PATCH_DRIVEDataLoader
from Unet_architecture import UNet2, UNet
from loss_functions import dice_loss, bce_loss, focal_loss






size = 128
train_transform = transforms.Compose([
    # transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([
    # transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 20
crop_size = (size,size)
trainset = PATCH_DRIVEDataLoader(train=True, transforms=train_transform,crop_size=crop_size )
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = PATCH_DRIVEDataLoader(train=False, transforms=test_transform, crop_size=crop_size)
test_loader = DataLoader(testset, shuffle=False, num_workers=3)


print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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


    
    
model = UNet2(200).to(device)
summary(model, (3, 128, 128))

torch.cuda.empty_cache()
train(model, optim.Adam(model.parameters(), 0.00001, weight_decay=0.0001), dice_loss, 250, train_loader, test_loader)



accuracy = 0.

predictions = predict2(model, test_loader)
print(f"Predictions: {predictions}")
print(predictions.sum() / len(train_loader))


# We save our compiled model. 
# This way we don't have to have the exact model defined, when we load it later
model_scripted = torch.jit.script(model)
model_scripted.save("model.pt")