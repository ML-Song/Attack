#coding=utf-8
import os
import tqdm
import torch
import numpy as np
from torch import nn
import torchvision as tv
from sklearn import metrics
import torchvision.utils as vutils
from torch.nn import functional as F

from utils import converter


class AttackNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.noise = nn.Parameter(torch.rand(size) / 100)
        
    def forward(self, x):
        return self.noise + x
        
        
class Attack(object):
    def __init__(self, classifier, img_size=(299, 299), transfrom=None, devices=[]):
        self.devices = devices
        self.img_size = img_size
        
        if transfrom is None:
            self.transfrom = tv.transforms.Compose([
                tv.transforms.Resize(img_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transfrom = transfrom
            
        self.classifier_single = classifier
        for i in self.classifier_single:
            i.eval()

        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.classifier = [i.cuda() for i in self.classifier_single]
        else:
            raise Exception('Multi Devices Not Supported!')
            
    def get_loss(self, pred, target, noise_img, img, targeted=True, weight=64, margin=0.1):
        target_one_hot = torch.zeros_like(pred[0])
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        loss_cls = []
#         if targeted:
#             loss_cls = [F.cross_entropy(i, target) for i in pred]
#         else:
#             loss_cls = [1 - F.cross_entropy(i, target) for i in pred]
        for p in pred:
            prob = F.softmax(p, dim=1)
            prob_gt = prob.gather(1, target.view(-1, 1))
            prob_max = prob.clone()
            prob_max[target_one_hot.type(torch.uint8)] = 0
            prob_max = prob_max.max(dim=1)[0]
            if targeted:
                prob_delta = prob_gt - prob_max - margin
            else:
                prob_delta = prob_max - prob_gt - margin
            loss_cls.append(-torch.clamp(prob_delta, max=0))
            
        noise_img_uint8 = (noise_img * 0.5 + 0.5) * 255
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        img_uint8 = (img * 0.5 + 0.5) * 255
        img_uint8 = F.interpolate(img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        loss_noise = F.mse_loss(noise_img_uint8, img_uint8)
        loss_noise = torch.sqrt(loss_noise)
        return sum(loss_cls) / len(loss_cls), loss_noise
        
    def predict(self, img, label, targeted, max_iteration=1000, margin=0.1):
        label_tensor = torch.tensor(label)
        original_x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        model_single = AttackNet(original_x.shape)
        opt = torch.optim.Adam(model_single.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_iteration)
        if len(self.devices) == 0:
            model = model_single
        elif len(self.devices) == 1:
            model = model_single.cuda()
            label_tensor = label_tensor.cuda()
        else:
            raise Exception('Multi Devices Not Supported!')
            
        for i in range(max_iteration):
            opt.zero_grad()
            out = model(original_x)
            out = F.interpolate(out, (224, 224), mode='bilinear', align_corners=True)
            cls = [i(out) for i in self.classifier]
            loss_cls, loss_noise = self.get_loss(cls, label_tensor, out, original_x, targeted=targeted, margin=margin)
            loss = loss_cls# + loss_noise
            if i % (max_iteration / 10) == 0:
                print([j.argmax(dim=1) for j in cls], loss_cls, loss_noise)
            loss.backward()
            opt.step()
            scheduler.step(i)
        noise_img = model(original_x)
        noise_img_uint8 = converter.float32_to_uint8(noise_img)
        noise_img_uint8 = noise_img_uint8.detach()
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
        noise_img_uint8 = noise_img_uint8.cpu().numpy().astype(np.uint8)
        noise_img_uint8 = np.transpose(noise_img_uint8, (0, 2, 3, 1))
        return noise_img_uint8