import torch
import torch.nn as nn
import torch.nn.functional as F


parameters = 64


    

class UNet2(nn.Module):
    def __init__(self, parameter_count = 100):
        super().__init__()

        def parameters_from_depth(parameters, depth):
            return parameters*2*depth
        
        par0 = parameters_from_depth(parameters, 1)
        par1 = parameters_from_depth(parameters, 2)
        par2 = parameters_from_depth(parameters, 3)
        par3 = parameters_from_depth(parameters, 4)
        par4 = parameters_from_depth(parameters, 5)
        
        
        # encoder (downsampling)
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, par0, 3, padding=1),
            nn.BatchNorm2d(par0),
            nn.ReLU(),
            nn.Conv2d(par0, par0, 3,stride=1, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par0),
            nn.ReLU(),
        )
        self.pool0 = nn.Conv2d(par0, par1, 3,stride=2, padding=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(par1, par1, 3, padding=1),
            nn.BatchNorm2d(par1),
            nn.ReLU(),
            nn.Conv2d(par1, par1, 3,stride=1, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par1),
            nn.ReLU(),
        )

        self.pool1 = nn.Conv2d(par1, par2, 3,stride=2, padding=1)   # 128 -> 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(par2, par2, 3, padding=1),
            nn.BatchNorm2d(par2),
            nn.ReLU(),
            nn.Conv2d(par2, par2, 3,stride=1, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par2),
            nn.ReLU(),
        )
        self.pool2 = nn.Conv2d(par2, par3, 3,stride=2, padding=1)   # 64 -> 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(par3, par3, 3, padding=1),
            nn.BatchNorm2d(par3),
            nn.ReLU(),
            nn.Conv2d(par3, par3, 3,stride=1, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par3),
            nn.ReLU(),
        )
        self.pool3 = nn.Conv2d(par3, par4, 3,stride=2, padding=1)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(par4, par4, 3, padding=1),
            nn.BatchNorm2d(par4),
            nn.ReLU(),
            nn.Conv2d(par4, par4, 3, padding=1),
            nn.BatchNorm2d(par4),
            nn.ReLU(),
            nn.Conv2d(par4, par4, 3, padding=1),
            nn.BatchNorm2d(par4),
            nn.ReLU(),
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(par4, par3, kernel_size=2, stride=2, padding=0) # 16 -> 32
        self.dec0 = nn.Sequential(
            nn.Conv2d(par3*2, par3, 3, padding=1),
            nn.BatchNorm2d(par3),
            nn.ReLU(),
            nn.Conv2d(par3, par3, 3, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par3),
            nn.ReLU(),
        )
        
        
        self.upsample1 = nn.ConvTranspose2d(par3, par2, kernel_size=2, stride=2, padding=0)  # 32 -> 64
        self.dec1 = nn.Sequential(
            nn.Conv2d(par2*2, par2, 3, padding=1),
            nn.BatchNorm2d(par2),
            nn.ReLU(),
            nn.Conv2d(par2, par2, 3, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par2),
            nn.ReLU(),
        )
        self.upsample2 = nn.ConvTranspose2d(par2, par1, kernel_size=2, stride=2, padding=0)  # 64 -> 128
        self.dec2 = nn.Sequential(
            nn.Conv2d(par1*2, par1, 3, padding=1),
            nn.BatchNorm2d(par1),
            nn.ReLU(),
            nn.Conv2d(par1, par1, 3, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par1),
            nn.ReLU(),
        )
        self.upsample3 = nn.ConvTranspose2d(par1, par0, kernel_size=2, stride=2, padding=0)  # 128 -> 256
        self.dec3 = nn.Sequential(
            nn.Conv2d(par0*2, par0, 3, padding=1),
            nn.BatchNorm2d(par0),
            nn.ReLU(),
            nn.Conv2d(par0, par0, 3, padding=1),  # 256 -> 128
            nn.BatchNorm2d(par0),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(par0, 1, 3, padding=1)

    def forward(self, x):
        # print("x: ", x.shape)
        
        # encoder
        x1 = self.layer0(x)
        e0 = self.pool0(x1) # 256 -> 128
        # print("e0: ", e0.shape)
        e1 = self.pool1(self.layer1(e0)) # 128 -> 64
        # print("e1: ", e1.shape)
        
        e2 = self.pool2(self.layer2(e1)) # 64 -> 32
        # print("e2: ", e2.shape)

        e3 = self.pool3(self.layer3(e2)) # 32 -> 16
        # print("e3: ", e3.shape)

        # bottleneck
        b = self.bottleneck_conv(e3)
        # print("b: ", b.shape)

        # decoder
         # decoder
        # print("d0: ", d0.shape)
        d0 = self.upsample0(b)  # Upsample to # 16 -> 32
        d0 = torch.cat([d0, e2], dim=1)  # Concatenate with e2 (32x32)
        d0 = self.dec0(d0)

        d1 = self.upsample1(d0)  # Upsample to # 32 -> 64
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate with e1 (64x64)
        d1 = self.dec1(d1)

        d2 = self.upsample2(d1)  # Upsample to # 64 -> 128
        d2 = torch.cat([d2, e0], dim=1)  # Concatenate with e0 (128x128)
        d2 = self.dec2(d2)

        
        
        d3 = self.upsample3(d2)  # Upsample to # 128 -> 256
        d3 = torch.cat([d3, x1], dim=1)  # Concatenate with e0 (256x256)
        d3 = self.dec3(d3)
        d3 = self.final(d3)
        return d3
    
    
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