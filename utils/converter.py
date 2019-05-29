#coding=utf-8
import torch


def float32_to_uint8(inputs):
    x = torch.clamp(inputs, min=-1, max=1)
    x = ((x * 0.5) + 0.5) * 255
    x = torch.round(x)
    return x


def uint8_to_float32(inputs):
    x = torch.clamp(inputs, min=0, max=255)
    x = ((x / 255) - 0.5) * 2
    return x
