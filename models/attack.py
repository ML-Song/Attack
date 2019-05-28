#coding=utf-8
import os
import tqdm
import torch
import numpy as np
from torch import nn
import torchvision as tv
import torchvision.utils as vutils
from torch.nn import functional as F

from utils import converter


class AttackNet(nn.Module):
    def __init__(self, backbone, out_channels=3, max_perturbation=32., scale=10):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.max_perturbation = max_perturbation
        self.scale = scale
        
    def forward(self, x, label, target):
        n, c, h, w = x.shape
        out = self.backbone(x)
        out = out.view(n, 2, -1, h, w)
        
        mask = out[:, 0]
        perturbation = out[:, 1]
        
        mask = mask.view(n, -1, self.out_channels, h, w)
        perturbation = perturbation.view(n, -1, self.out_channels, h, w)
        
        label_index = label.view(-1, 1, 1, 1, 1)
        label_index = label_index.repeat(1, 1, self.out_channels, h, w)
        
        target_index = target.view(-1, 1, 1, 1, 1)
        target_index = target_index.repeat(1, 1, self.out_channels, h, w)

        mask = mask.gather(1, label_index).squeeze(dim=1)
        mask = torch.sigmoid(mask)
#         mask_min = mask.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
#         mask_max = mask.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
#         mask = (mask - mask_min) / (mask_max - mask_min)
#         mask = torch.sigmoid(self.scale * (mask - 0.5))
        
        perturbation = perturbation.gather(1, target_index).squeeze(dim=1)
#         perturbation = perturbation / ((128 / self.max_perturbation) * torch.sqrt((perturbation ** 2).sum(dim=1, keepdim=True)))
        perturbation_min = perturbation.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        perturbation_max = perturbation.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        perturbation = ((perturbation - perturbation_min) / (perturbation_max - perturbation_min) - 0.5) * 2
        perturbation = perturbation * (self.max_perturbation / 128)
        return mask, perturbation
        
        
class Attack(object):
    def __init__(self, net, classifier=None, train_loader=None, test_loader=None, batch_size=None, gain=None, 
                 optimizer='sgd', lr=1e-3, patience=5, interval=1, weight=64, 
                 img_size=(299, 299), transfrom=None, loss_mode='cross_entropy', margin=0.5, 
                 checkpoint_dir='saved_models', checkpoint_name='', devices=[0], targeted=True, num_classes=110):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.devices = devices
        self.targeted = targeted
        self.img_size = img_size
        self.num_classes = num_classes
        self.loss_mode = loss_mode
        assert(self.loss_mode in ('margin', 'cross_entropy'))
        self.weight = weight
        self.margin = margin
        
        if transfrom is None:
            self.transfrom = tv.transforms.Compose([
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transfrom = transfrom
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        self.net_single = net
        self.classifier_single = classifier
        if isinstance(classifier, list):
            for i in self.classifier_single:
                i.eval()
                i.volatile = True

        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.net = self.net_single.cuda()
            if isinstance(classifier, list):
                self.classifier = [i.cuda() for i in self.classifier_single]
        else:
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).cuda()
            if isinstance(classifier, list):
                self.classifier = [nn.DataParallel(i, device_ids=range(len(devices))).cuda() for i in self.classifier_single]
            
        if gain is not None:
            self.gain = gain
            
        if optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                self.net_single.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
        elif optimizer == 'adam':
            self.opt = torch.optim.Adam(
                self.net_single.parameters(), lr=lr, weight_decay=5e-4)
        else:
            raise Exception('Optimizer {} Not Exists'.format(optimizer))
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None, epoch_size=10, mode='perturbation'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_epoch * epoch_size)
        torch.cuda.manual_seed(1)
        best_score = 255
        step = 1
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            self.net.train()
            for batch_idx, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                if self.targeted and mode == 'perturbation':
                    if len(data) == 2:
                        target = torch.randint_like(label, 0, self.num_classes).cuda()
                    else:
                        target = data[2].cuda()
                else:
                    target = label

                self.reset_grad()
                mask, perturbation = self.net(img, label, target)
                img_mean = img.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                if self.targeted:
                    if mode == 'mask':
                        img_masked = img * mask + img_mean * (1 - mask)
                        perturbated_img = img_masked
                    else:
                        img_masked = img# * mask + img_mean * (1 - mask)
                        perturbated_img = img_masked + perturbation
                else:
                    perturbated_img = img + perturbation
#                 if self.targeted:
#                     perturbation = perturbation * mask
                
                cls = [c(perturbated_img) for c in self.classifier]
                
                loss_cls, loss_perturbation = self.get_loss(cls, target if self.targeted else label, 
                                                            perturbated_img, img, self.targeted and mode == 'perturbation')
                
                loss = loss_cls + loss_perturbation
                
                loss.backward()
                self.opt.step()
                scheduler.step(step)
                if writer:
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=step)
                    writer.add_scalar(
                        'loss_cls', loss_cls.data, global_step=step)
#                     for i, l in enumerate(loss_cls):
#                         writer.add_scalar(
#                             'loss_cls_{}'.format(i), l.data, global_step=step)
                    writer.add_scalar(
                        'loss_perturbation', loss_perturbation.data, global_step=step)
                    writer.add_scalar(
                        'loss', loss.data, global_step=step)
                step += 1
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                acc, perturbation, imgs, perturbated_imgs = self.test(self.targeted, mode)
                if writer:
                    for i, (a, p) in enumerate(zip(acc, perturbation)):
                        writer.add_scalar(
                            'acc_{}'.format(i), a, global_step=epoch)
                        writer.add_scalar(
                            'perturbation_{}'.format(i), p, global_step=epoch)
                        
                    acc_mean = sum(acc) / len(acc)
                    perturbation_mean = sum(perturbation) / len(perturbation)
                    if self.targeted and mode == 'perturbation':
                        score = (1 - acc_mean) * 64 + perturbation_mean
                    else:
                        score = acc_mean * 64 + perturbation_mean
                    
                    writer.add_scalar(
                            'acc_mean', acc_mean, global_step=epoch)
                    writer.add_scalar(
                            'perturbation_mean', perturbation_mean, global_step=epoch)
                    writer.add_scalar(
                            'score', score, global_step=epoch)
                    imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
                    writer.add_image('Imgs', imgs, epoch)
                    perturbated_imgs = vutils.make_grid(perturbated_imgs, normalize=True, scale_each=True)
                    writer.add_image('Perturbated Imgs', perturbated_imgs, epoch)
                    
                if best_score > score:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self, targeted=False, mode='mask'):
        self.net.eval()
        with torch.no_grad():
            preds = []
            gt = []
            perturbations = []
            imgs = []
            noise_imgs = []
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                if targeted and mode == 'perturbation':
                    if len(data) == 2:
                        target = torch.randint_like(label, 0, self.num_classes).cuda()
                    else:
                        target = data[2].cuda()
                    gt.append(target.cpu())
                else:
                    target = label.cuda()
                    gt.append(label.cpu())
                    
                mask, noise = self.net(img, label, target)
                img_mean = img.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                if targeted:
                    if mode == 'mask':
                        img_masked = img * mask + img_mean * (1 - mask)
                        noise_img = img_masked
                    else:
                        img_masked = img * mask + img_mean * (1 - mask)
                        noise_img = img_masked + noise
                else:
                    noise_img = img + noise

                noise_img_uint8 = converter.float32_to_uint8(noise_img)
                noise_img = converter.uint8_to_float32(noise_img_uint8)
                
                if batch_idx < 4:
                    imgs.append(img.cpu())
                    noise_imgs.append(noise_img.cpu())
                
                img_uint8 = converter.float32_to_uint8(img)

                perturbation = noise_img_uint8 - img_uint8
                perturbation = F.interpolate(perturbation, self.img_size, mode='bilinear', align_corners=True)
                perturbation = torch.sqrt((perturbation ** 2).sum(dim=1)).mean(dim=-1).mean(dim=-1)
                perturbations.append(perturbation.detach().cpu())
                
                cls = [c(noise_img) for c in self.classifier]
                preds.append([c.argmax(dim=1).detach().cpu() for c in cls])
                
            preds = [torch.cat([p[i] for p in preds]).numpy() for i, c in enumerate(self.classifier)]
            gt = torch.cat(gt).numpy()
            perturbations = torch.cat(perturbations).numpy()
            imgs = torch.cat(imgs)
            noise_imgs = torch.cat(noise_imgs)
            acc = [(gt == p).astype(np.float32).mean() for p in preds]
            
            if targeted and mode == 'perturbation':
                perturbation = [perturbations[gt == p].sum() / len(perturbations) if (gt == p).sum() > 0 else 0 for p in preds]
            else:
                perturbation = [perturbations[gt != p].sum() / len(perturbations) if (gt != p).sum() > 0 else 0 for p in preds]
        return acc, perturbation, imgs, noise_imgs

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single.state_dict(), '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single.state_dict(), '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path))
    
    def predict(self, img, label, target):
        x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        label_tensor = torch.tensor(label).cuda()
        target_tensor = torch.tensor(target).cuda()
        self.net.eval()
        with torch.no_grad():
            mask, noise = self.net(x, label_tensor, target_tensor)
            img_mean = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            img_masked = x * mask + img_mean * (1 - mask)
            noise_img = img_masked + noise
            noise_img_uint8 = converter.float32_to_uint8(noise_img)
            noise_img_uint8 = noise_img_uint8.detach()
            noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
            noise_img_uint8 = noise_img_uint8.cpu().numpy().astype(np.uint8)
            noise_img_uint8 = np.transpose(noise_img_uint8, (0, 2, 3, 1))
        return noise_img_uint8
    
    def get_loss(self, preds, target, noise_img, img, targeted, max_perturbation=10):
        loss_perturbation = F.mse_loss(noise_img, img)
        loss_cls_non_target = 0
        for out in preds:
            out_inverse = torch.log(torch.clamp(1 - torch.softmax(out, dim=1), min=1e-6))
            loss_cls_non_target = loss_cls_non_target + F.nll_loss(out_inverse, target)
#         loss = (loss_cls_non_target + beta * loss_min_noise) / (1 + beta)
        return loss_cls_non_target, 8 * loss_perturbation
        
        
        
        target_one_hot = torch.zeros_like(preds[0])
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        cls_losses = []
        perturbation_losses = []
        
        noise_img_uint8 = noise_img * 128
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)

        img_uint8 = img * 128
        img_uint8 = F.interpolate(img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
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