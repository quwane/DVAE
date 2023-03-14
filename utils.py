# coding : UTF - 8
# Author : Ziyu Zhan
# Creation Date : 2022/10/14
# File : utils.py

import torch
import numpy as np
import torch.fft as fourier
import torch.nn as nn
import math
import os
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(40)

def transfer_kernel(z, wavelength, N, pixel_size, bandlimit=True, gpu=True):
    """
    Generate a propagation kernel (with zero padding).

    Parameters
    ----------
    z : Propagation distance
    wavelength : optical frequency
    N: resolution (before zero padding)
    pixel_size: as it implies
    bandlimit: whether to use limited band. Default: True
    gpu: whether cuda or not. Default: True

    Returns
    -------
    h : tensor
        The propagation kernel."""

    k = 2.0 * np.pi / wavelength
    length_screen = pixel_size * N
    fs = 1.0 / (2 * length_screen) # '2' comes from zero padding
    fx = np.linspace(-1/(2*pixel_size), (1/(2*pixel_size) - fs), 2 * N)
    fy = fx
    Fx, Fy = np.meshgrid(fx, fy)
    ph0 = Fx ** 2 + Fy ** 2
    ph = np.exp(1.0j * z * np.sqrt(k ** 2 - np.multiply(4 * np.pi ** 2, ph0)))
    if bandlimit:
        fxlimit = 1 / np.sqrt(1 + (2 * fs * z) ** 2) / wavelength
        fylimit = fxlimit
        ph[np.abs(Fx) > fxlimit] = 0
        ph[np.abs(Fy) > fylimit] = 0

    h = np.fft.fftshift(ph)
    h = torch.from_numpy(h)
    if gpu:
        h = h.cuda()
    return h

def diffraction(wave, trans_func):
    padding_size = wave.shape[1] // 2
    wave = F.pad(wave, pad=[padding_size, padding_size, padding_size, padding_size])
    wave = torch.squeeze(wave)
    wave_f = fourier.fft2(fourier.fftshift(wave))
    wave_f *= trans_func
    wave = fourier.ifftshift(fourier.ifft2(wave_f))
    wave = wave[:, wave.shape[1] // 4: (wave.shape[1] // 4 + 2 * padding_size), wave.shape[1] // 4: (wave.shape[1] // 4 + 2 * padding_size)]
    return wave

def modulation(wave, plate, mode):
    """
    Optical Modulation Layer.

    :param wave: Input wave, type: tensor
    :param plate: modulation layer, type: tensor
    :param mode: modulation mode, type: str, specify 'mode1' or 'mode2'
    :return: Output wave after modulation, type: tensor
    """

    if strcmp(mode, 'mode1'):
        scale = 10
        alpha = 2
        wave_m = wave * torch.exp(1.0j * scale * math.pi * (torch.sin(alpha * plate) + 1))
    elif strcmp(mode, 'mode2'):
        wave_m = wave * torch.exp(1.0j * 2 * math.pi * torch.sigmoid(plate))
    # wave_mf = wave_m[:, paddings: paddings + 200, paddings: paddings + 200]
    return wave_m

def reparametrize(self, mu, sigma):
    std = torch.exp(sigma / 2)
    eps = torch.randn_like(std)
    return mu + eps * std
