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
        self.noise = nn.Parameter(torch.zeros(size))
        
    def forward(self, x):
        out = self.noise + x
        out = torch.clamp(out, min=-1, max=1)
        return out
    
    def renorm_grad(self, lr, levels=1/128):
        max_grad = levels / self.noise.grad.abs().max()
        self.noise.grad *= max_grad
        self.noise.grad[self.noise.grad.abs() < levels / lr] = 0
        
        
class Attack(object):
    def __init__(self, classifiers, input_size=(224, 224), output_size=(299, 299), 
                 transfrom=None, device='cpu'):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        if transfrom is None:
            self.transfrom = tv.transforms.Compose([
                tv.transforms.Resize(output_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transfrom = transfrom
            
        self.classifiers_single = classifiers
        for i in self.classifiers_single:
            i.eval()

        if device == 'cpu':
            pass
        elif device == 'cuda':
            self.classifiers = [i.cuda() for i in self.classifiers_single]
        else:
            raise Exception('Device {} Not Found!'.format(device))
            
    def get_loss(self, preds, target, noise_img, img, targeted=True, max_perturbation=20):
        target_one_hot = torch.zeros_like(preds[0])
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        cls_losses = []
        perturbation_losses = []
        
        for p in preds:
            if targeted:
                loss_cls = F.cross_entropy(p, target, reduction='none')
                loss_cls = loss_cls[p.argmax(dim=-1) != target].sum() / target.size(0)
            else:
                loss_cls = -F.cross_entropy(p, target, reduction='none')
                loss_cls = loss_cls[p.argmax(dim=-1) == target].sum() / target.size(0)
            cls_losses.append(loss_cls)
            
            noise_img_uint8 = (noise_img * 0.5 + 0.5) * 255
            noise_img_uint8 = F.interpolate(noise_img_uint8, self.output_size, mode='bilinear', align_corners=True)
        
            img_uint8 = (img * 0.5 + 0.5) * 255
            img_uint8 = F.interpolate(img_uint8, self.output_size, mode='bilinear', align_corners=True)

            loss_perturbation = F.mse_loss(noise_img_uint8, img_uint8, reduction='none')
            loss_perturbation = torch.sqrt(loss_perturbation.sum(dim=1))
            loss_perturbation = loss_perturbation.mean()
#             if targeted:
#                 loss_perturbation = loss_perturbation[p.argmax(dim=-1) == target].sum() / target.size(0)
#             else:
#                 loss_perturbation = loss_perturbation[p.argmax(dim=-1) != target].sum() / target.size(0)
            perturbation_losses.append(loss_perturbation)
        return sum(cls_losses) / len(cls_losses), sum(perturbation_losses) / len(perturbation_losses)
        
    def predict(self, img, label, targeted, max_iteration=1000, max_perturbation=20, lr=3):
        label_tensor = torch.tensor(label)
        original_x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        model_single = AttackNet(original_x.shape)
        opt = torch.optim.SGD(model_single.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_iteration)
        if self.device == 'cpu':
            model = model_single
        elif self.device == 'cuda':
            model = model_single.cuda()
            label_tensor = label_tensor.cuda()
        else:
            raise Exception('Device {} Not Found!'.format(device))
            
        for step in range(max_iteration):
            opt.zero_grad()
            out = model(original_x)
            out = F.interpolate(out, self.input_size, mode='bilinear', align_corners=True)
            cls = [i(out) for i in self.classifiers]
            loss_cls, loss_perturbation = self.get_loss(cls, label_tensor, 
                                                 out, original_x, targeted=targeted, 
                                                 max_perturbation=max_perturbation)
            loss = loss_cls + loss_perturbation
            if step % (max_iteration / 10) == 0:
                print([j.argmax(dim=1).data for j in cls], loss_cls.data, loss_perturbation.data)
            loss.backward()
#             model.renorm_grad(lr=lr)
            opt.step()
            scheduler.step(step)
            
        noise_img = model(original_x)
        noise_img_uint8 = converter.float32_to_uint8(noise_img)
        noise_img_uint8 = noise_img_uint8.detach()
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.output_size, mode='bilinear', align_corners=True)
        noise_img_uint8 = noise_img_uint8.cpu().numpy().astype(np.uint8)
        noise_img_uint8 = np.transpose(noise_img_uint8, (0, 2, 3, 1))
        return noise_img_uint8