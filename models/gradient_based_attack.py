#coding=utf-8
import os
import cv2
import tqdm
import torch
import numpy as np
from torch import nn
from PIL import Image
import torchvision as tv
from sklearn import metrics
import torchvision.utils as vutils
from torch.nn import functional as F

from utils import converter


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    g = np.multiply(kx, np.transpose(ky))
    kernel = np.array([np.array([g, np.zeros_like(g), np.zeros_like(g)]), 
              np.array([np.zeros_like(g), g, np.zeros_like(g)]), 
              np.array([np.zeros_like(g), np.zeros_like(g), g])])
    return kernel.astype(np.float32)


class AttackNet(nn.Module):
    def __init__(self, size, init_mode='zero'):
        super().__init__()
        if init_mode == 'zero':
            self.noise = nn.Parameter(torch.zeros(size))
        elif init_mode == 'randn':
            self.noise = nn.Parameter(torch.randn(size) / 5)
        else:
            raise Exception('Init Mode {} Not Supported'.format(init_mode))
        
    def forward(self, x):
        out = self.noise + x
        out = torch.clamp(out, min=-1, max=1)
        return out
    
    def renorm_grad(self, lr, levels=1/256):
#         print(self.noise.grad)
        max_grad = levels / torch.clamp(self.noise.grad.abs().max(), min=1e-6, max=1)
        self.noise.grad *= max_grad
#         self.noise.grad[self.noise.grad.abs() < levels / lr] = 0
#         self.noise.grad = torch.clamp(self.noise.grad, min=-1, max=1)
        
        
class Attack(object):
    def __init__(self, classifiers, test_classifiers=None, 
                 patience=5, max_iteration=10, input_size=(224, 224), output_size=(299, 299), 
                 transfrom=None, device='cuda', weight=64, max_epoch=10, loss_mode='margin', margin=0.1):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.patience = patience
        self.max_iteration = max_iteration
        self.weight = weight
        self.max_epoch = max_epoch
        self.loss_mode = loss_mode
        assert(self.loss_mode in ('margin', 'cross_entropy'))
        self.margin = margin
        
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
            self.classifiers = self.classifiers_single
        elif device == 'cuda':
            self.classifiers = [i.cuda() for i in self.classifiers_single]
        else:
            raise Exception('Device {} Not Found!'.format(device))
            
        if test_classifiers is not None:
            self.test_classifiers_single = test_classifiers
            for i in self.test_classifiers_single:
                i.eval()

            if device == 'cpu':
                self.test_classifiers = self.test_classifiers_single
            elif device == 'cuda':
                self.test_classifiers = [i.cuda() for i in self.test_classifiers_single]
            else:
                raise Exception('Device {} Not Found!'.format(device))
        else:
            self.test_classifiers = self.classifiers
            
    def get_loss(self, preds, target, noise_img, img, targeted=True, max_perturbation=20):
        target_one_hot = torch.zeros_like(preds[0])
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        cls_losses = []
        perturbation_losses = []
        
        noise_img_uint8 = noise_img * 128
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.output_size, mode='bilinear', align_corners=True)

        img_uint8 = img * 128
        img_uint8 = F.interpolate(img_uint8, self.output_size, mode='bilinear', align_corners=True)
        
        loss_perturbation = torch.clamp(((img_uint8 - noise_img_uint8) ** 2).sum(dim=1), min=1e-6)
        loss_perturbation = torch.sqrt(loss_perturbation)
        loss_perturbation = torch.clamp(loss_perturbation - max_perturbation, min=0).mean(dim=-1).mean(dim=-1)
#         print(loss_perturbation)
#         loss_perturbation = loss_perturbation.mean()
        for p in preds:
            if self.loss_mode == 'margin':
                prob = F.softmax(p, dim=1)
                prob_gt = prob.gather(1, target.view(-1, 1)).view(-1)
                prob_other_max = prob.clone()
                prob_other_max[target_one_hot.type(torch.uint8)] = -1
                prob_other_max = prob_other_max.max(dim=1)[0]
            if targeted:
                if self.loss_mode == 'margin':
                    prob_delta = prob_other_max - prob_gt + self.margin
                    loss_cls = torch.clamp(prob_delta, min=0,).sum() / target.size(0)
                elif self.loss_mode == 'cross_entropy':
                    loss_cls = F.cross_entropy(p, target, reduction='none')
                    loss_cls = loss_cls[p.argmax(dim=-1) != target].sum() / target.size(0)
                loss_noise = loss_perturbation[p.argmax(dim=-1) == target].sum() / target.size(0)
            else:
                if self.loss_mode == 'margin':
                    prob_delta = prob_gt - prob_other_max + self.margin
                    loss_cls = torch.clamp(prob_delta, min=0).sum() / target.size(0)
                elif self.loss_mode == 'cross_entropy':
                    loss_cls = -F.cross_entropy(p, target, reduction='none')
                    loss_cls = loss_cls[p.argmax(dim=-1) == target].sum() / target.size(0)
                loss_noise = loss_perturbation[p.argmax(dim=-1) != target].sum() / target.size(0)
#             loss_cls = loss_cls.sum() / loss_cls.size(0)
            cls_losses.append(loss_cls)
            perturbation_losses.append(loss_noise)
        return sum(cls_losses) / len(cls_losses) * self.weight, sum(perturbation_losses) / len(perturbation_losses)
        
    def update_one_step(self, original_x, label_tensor, targeted, lr=10, max_perturbation=20, use_post_process=False):
        for step in range(self.max_iteration):
            self.opt.zero_grad()
            out = self.model(original_x)
            out = F.interpolate(out, self.input_size, mode='bilinear', align_corners=True)
            cls = [c(out) for c in self.classifiers]
            loss_cls, loss_perturbation = self.get_loss(cls, label_tensor, 
                                                 out, original_x, targeted=targeted, 
                                                 max_perturbation=max_perturbation)
            loss = loss_perturbation + loss_cls
            if use_post_process:
                kernel_size = 5
                kernel = torch.tensor(gaussian_kernel_2d_opencv(kernel_size))
                if self.device == 'cuda':
                    kernel = kernel.cuda()
                out_processed = F.conv2d(out, kernel, padding=kernel_size//2)
                
                cls_processed = [i(out_processed) for i in self.classifiers]
                loss_cls_processed, loss_perturbation_processed = self.get_loss(cls_processed, label_tensor, 
                                                                                out, original_x, targeted=targeted, 
                                                                                max_perturbation=max_perturbation)
                loss = loss + loss_cls_processed + loss_perturbation_processed
#             if step % (max_iteration / 10) == 0:
#                 print([j.argmax(dim=1).data for j in cls], loss_cls.data, loss_perturbation.data)
#             print(loss)
            loss.backward()
            self.model.renorm_grad(lr=lr)
            self.opt.step()
            
        noise_img = self.model(original_x)
        noise_img_uint8 = converter.float32_to_uint8(noise_img)
        noise_img_uint8 = noise_img_uint8.detach()
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.output_size, mode='bilinear', align_corners=True)
        noise_img_uint8 = noise_img_uint8.cpu().numpy().astype(np.uint8)
        noise_img_uint8 = np.transpose(noise_img_uint8, (0, 2, 3, 1))
        
        noise_img_pil = [Image.fromarray(i) for i in noise_img_uint8]
        return noise_img_pil
    
    def get_score(self, noise_img, img, label, targeted):
        label_tensor = torch.tensor(label)
        noise_img = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in noise_img], dim=0).cuda()
        original_img = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0)
            
        noise_cls = [c(F.interpolate(noise_img, 
                                     self.input_size, 
                                     mode='bilinear', 
                                     align_corners=True)).detach().cpu() for c in self.test_classifiers]
        noise_img = noise_img.cpu()
        if targeted:
            acc = [(c.argmax(dim=1) != label_tensor).type(torch.float32).mean() for c in noise_cls]
        else:
            acc = [(c.argmax(dim=1) == label_tensor).type(torch.float32).mean() for c in noise_cls]
        l2_distance = converter.float32_to_uint8(noise_img) - converter.float32_to_uint8(original_img)
        l2_distance = torch.sqrt((l2_distance ** 2).sum(dim=1)).mean()
        print(acc, l2_distance)
        return sum(acc) / len(acc) * self.weight + l2_distance#, acc, l2_distance
    
    def predict(self, img, label, targeted=False, lr=10, max_perturbation=0):
        original_x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        label_tensor = torch.tensor(label)
        self.model_single = AttackNet(original_x.shape, 'zero')
        if self.device == 'cpu':
            self.model = self.model_single
        elif self.device == 'cuda':
            self.model = self.model_single.cuda()
            label_tensor = label_tensor.cuda()
        else:
            raise Exception('Device {} Not Found!'.format(device))
        self.opt = torch.optim.SGD(self.model_single.parameters(), lr=lr, momentum=0.99)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.max_epoch)
        used_patience = 0
        best_score = 255
        best_result = None
        for epoch in range(self.max_epoch):
            result = self.update_one_step(original_x, label_tensor, targeted, lr, max_perturbation=max_perturbation)
            score = self.get_score(result, img, label, targeted)
            print(score)
            if score < best_score:
                best_score = score
                best_result = result
                used_patience = 0
            else:
                used_patience += 1
                
            if used_patience >= self.patience:
#                 pass
                break
#             self.scheduler.step(epoch)
        return best_result