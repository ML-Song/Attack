#coding=utf-8
import torch
import numpy as np
import torchvision as tv 

import sys
sys.path.append('../')
import unet
from .converter import float32_to_uint8, uint8_to_float32
import models.gain as gain


class GAINAPI(object):
    def __init__(self, model_path=None, num_classes=110):
        mean_arr = [0.5, 0.5, 0.5]
        stddev_arr = [0.5, 0.5, 0.5]
        normalize = tv.transforms.Normalize(mean=mean_arr,
                                         std=stddev_arr)

        model_dimension = 224
        center_crop = 224
        self.data_transform = tv.transforms.Compose([
            tv.transforms.Resize(model_dimension),
            tv.transforms.CenterCrop(center_crop),
            tv.transforms.ToTensor(),
            normalize,
        ])
        self.model = gain.GAIN(num_classes).cuda()
            
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.softmask = gain.SoftMask()
        
    def __call__(self, img):
        with torch.no_grad():
            x = self.data_transform(img).unsqueeze(0).cuda()
            n, c, h, w = x.shape
            out, out_masked, cam = self.model(x)
            cam = cam.cpu().squeeze().numpy()
        return cam
    
    
class AttackAPI(object):
    def __init__(self, model_path=None, with_target=False, num_classes=110):
        mean_arr = [0.5, 0.5, 0.5]
        stddev_arr = [0.5, 0.5, 0.5]
        normalize = tv.transforms.Normalize(mean=mean_arr,
                                         std=stddev_arr)

        model_dimension = 224
        center_crop = 224
        self.data_transform = tv.transforms.Compose([
#             tv.transforms.ToPILImage(), 
            tv.transforms.Resize(model_dimension),
            tv.transforms.CenterCrop(center_crop),
            tv.transforms.ToTensor(),
            normalize,
        ])
        if with_target:
            self.model = unet.UNet(3, 3 * num_classes, batch_norm=True).cuda()
        else:
            self.model = unet.UNet(3, 3, batch_norm=True).cuda()
            
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def __call__(self, img, target=None):
        with torch.no_grad():
            x = self.data_transform(img).unsqueeze(0).cuda()
            n, c, h, w = x.shape
            noise = self.model(x)
            if target is not None:
                noise = noise.view(n, -1, c, h, w)
                noise = noise[0, target]
            x = x + noise
            x = float32_to_uint8(x)
            x = torch.clamp(x, min=0, max=255).cpu().squeeze().numpy()
        return np.transpose(x, (1, 2, 0)).astype(np.uint8)