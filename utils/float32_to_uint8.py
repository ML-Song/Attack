#coding=utf-8
import torch


def float32_to_uint8(x):
    x = torch.clamp(x, min=-1, max=1)
    x = ((x * 0.5) + 0.5) * 255
    x = torch.floor(x)
    x = ((x / 255) - 0.5) * 2
    return x
