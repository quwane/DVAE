# Coding: UTF-8
# Author Ziyu Zhan
# Creation Date: 2021/12/27
# VAE of D2NN
#  ============================
import torch
import numpy as np
import torch.fft as fourier
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from utils import transfer_kernel, diffraction, modulation, reparametrize

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# global parameters
size = 200
layer = 4
distance = 40
ls = 106.7 # length of screen
wl = 1 # wavelength
k = 2.0 * math.pi / wl
electric_input = 100
electric_dim1 = 1000
electric_output = 2500
paddings = 100
kernel_size = 10
stride = 10
extension = 11.0
scale = 2.0
# diffraction propagation
h_interlayer = transfer_kernel(z=distance, wavelength=wl, N=n_num, pixel_size=pixel_size, bandlimit=True, gpu=True)


class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.phase1 = torch.nn.Parameter(np.sqrt(0.0 * math.pi) * torch.randn(num_layers, size, size), requires_grad=True)
        self.phase2 = torch.nn.Parameter(np.sqrt(0.0 * math.pi) * torch.randn(num_layers, size, size), requires_grad=True)
        self.fcl = nn.Sequential(
            nn.Linear(electric_input, electric_dim1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(electric_dim1, electric_output),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
        self.BN = nn.Sequential(nn.BatchNorm2d(1))
        self.avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, stride=stride))

    def forward(self, x):
        x1 = x
        for idx in range(self.num_layers):
            x2 = modulation(x, self.phase1[idx], 'mode2')
            x2 = diffraction(x2, h_interlayer)
            x2 = torch.abs(x2)  # 200*200
            x3 = modulation(x1, self.phase2[idx], 'mode2')
            x3 = diffraction(x3, h_interlayer)
            x3 = torch.abs(x3)  # 200*200
        ccd1 = torch.square(x2)
        ccd2 = torch.square(x3)
        ccd1 = ccd1[:, 50: 150, 50: 150]
        ccd2 = ccd2[:, 50: 150, 50: 150]
        mu = self.avg_pool(ccd1)
        sigma = self.avg_pool(ccd2)
        latent = reparametrize(mu, sigma)
        reconstruct = self.fcl(latent)
        return reconstruct, mu, sigma
