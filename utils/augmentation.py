#coding=utf-8
import cv2
import torch
import numpy as np
import torchvision as tv
from torch.nn import functional as F


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    g = np.multiply(kx, np.transpose(ky))
    kernel = np.array([np.array([g, np.zeros_like(g), np.zeros_like(g)]), 
              np.array([np.zeros_like(g), g, np.zeros_like(g)]), 
              np.array([np.zeros_like(g), np.zeros_like(g), g])])
    return kernel.astype(np.float32)


class RandomNoise(object):
    def __init__(self, scale=0.1):
        self.scale = scale
        
    def __call__(self, x):
        return x + torch.randn_like(x) * self.scale
    
    
class RandomBlur(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.kernel = torch.tensor(gaussian_kernel_2d_opencv(kernel_size))
        
    def __call__(self, x):
        return F.conv2d(x.unsqueeze(dim=0), self.kernel, padding=self.kernel_size//2).squeeze(dim=0)