#coding=utf-8
import cv2
import numpy as np


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    g = np.multiply(kx, np.transpose(ky))
    kernel = np.array([np.array([g, np.zeros_like(g), np.zeros_like(g)]), 
              np.array([np.zeros_like(g), g, np.zeros_like(g)]), 
              np.array([np.zeros_like(g), np.zeros_like(g), g])])
    return kernel.astype(np.float32)