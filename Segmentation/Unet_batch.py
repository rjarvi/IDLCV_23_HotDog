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


data_path = '/dtu/datasets1/02516/phc_data'
#data_path = './phc_data'
class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    
    
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 25
trainset = PhC(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = PhC(train=False, transform=test_transform)
test_loader = DataLoader(testset, shuffle=False, num_workers=3)


print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

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

parameters = 100
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, parameters, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(parameters, parameters, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(parameters, parameters, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(parameters, parameters, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(parameters, parameters, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(parameters*2, parameters, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(parameters*2, parameters, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(parameters*2, parameters, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(parameters*2, 1, 3, padding=1)

    def forward(self, x):
        # print("x: ", x.shape)
        
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x))) # 128 -> 64
        # print("e0: ", e0.shape)
        e1 = self.pool1(F.relu(self.enc_conv1(e0))) # 64 -> 32
        # print("e1: ", e1.shape)
        
        e2 = self.pool2(F.relu(self.enc_conv2(e1))) # 32 -> 16
        # print("e2: ", e2.shape)

        e3 = self.pool3(F.relu(self.enc_conv3(e2))) # 16 -> 8
        # print("e3: ", e3.shape)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        # print("b: ", b.shape)

        # decoder
         # decoder
        d0 = b  # This should be of shape [B, C, 16, 16]
        # print("d0: ", d0.shape)
        d0 = torch.cat([d0, e3], dim=1)  # Concatenate with e3
        d0 = F.relu(self.dec_conv0(d0))

        d1 = self.upsample0(d0)  # This should be of shape [B, C, 32, 32]
        # print("d1: ", d1.shape)
        
        d1 = torch.cat([d1, e2], dim=1)  # Concatenate with e2
        d1 = F.relu(self.dec_conv1(d1))

        d2 = self.upsample1(d1)  # This should be of shape [B, C, 64, 64]
        # print("d2: ", d2.shape)
        
        d2 = torch.cat([d2, e1], dim=1)  # Concatenate with e1
        d2 = F.relu(self.dec_conv2(d2))

        d3 = self.upsample2(d2)  # This should be of shape [B, C, 128, 128]
        # print("d3: ", d3.shape)
        
        d3 = torch.cat([d3, e0], dim=1)  # Concatenate with e0
        d3 = self.dec_conv3(d3)  # No activation
        d3 = self.upsample3(d3)
        
        return d3
    
    
model = UNet().to(device)
summary(model, (3, 128, 128))

torch.cuda.empty_cache()
train(model, optim.Adam(model.parameters()), bce_loss, 40, train_loader, test_loader)



accuracy = 0.

predictions = predict2(model, test_loader)
print(f"Predictions: {predictions}")
print(predictions.sum() / len(train_loader))


# We save our compiled model. 
# This way we don't have to have the exact model defined, when we load it later
model_scripted = torch.jit.script(model)
model_scripted.save("model.pt")