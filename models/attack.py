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
    def __init__(self, backbone, out_channels=3):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        
    def forward(self, x, label, target):
        n, c, h, w = x.shape
        out = self.backbone(x)
        out = out.view(n, 2, -1, h, w)
        
        mask = out[:, 0]
        noise = out[:, 1]
        
        mask = mask.view(n, -1, self.out_channels, h, w)
        noise = noise.view(n, -1, self.out_channels, h, w)
        
        label_index = label.view(-1, 1, 1, 1, 1)
        label_index = label_index.repeat(1, 1, self.out_channels, h, w)
        
        target_index = target.view(-1, 1, 1, 1, 1)
        target_index = target_index.repeat(1, 1, self.out_channels, h, w)

        mask = mask.gather(1, label_index).squeeze(dim=1)
#         mask = torch.sigmoid(mask)
        mask_min = mask.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        mask_max = mask.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        mask = (mask - mask_min) / (mask_max - mask_min)
        mask = torch.sigmoid(5 * (mask - 0.5))
        
        noise = noise.gather(1, target_index).squeeze(dim=1)
        noise_min = noise.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        noise_max = noise.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        noise = ((noise - noise_min) / (noise_max - noise_min) - 0.5) * 2
        return mask, noise
#         if target is None:
#             return mask, noise
#         else:
#             index = target.view(-1, 1, 1, 1, 1)
#             index = index.repeat(1, 1, self.out_channels, h, w)
            
#             mask = mask.view(n, -1, self.out_channels, h, w)
#             mask = mask.gather(1, index).squeeze(dim=1)
            
#             noise = noise.view(n, -1, self.out_channels, h, w)
#             noise = noise.gather(1, index).squeeze(dim=1)
#             mask = torch.cat([mask[i, label[i]].unsqueeze(0) for i in range(n)], dim=0)
#             noise = torch.cat([noise[i, target[i]].unsqueeze(0) for i in range(n)], dim=0)
#             return mask, noise
        
        
class Attack(object):
    def __init__(self, net, classifier=None, train_loader=None, test_loader=None, batch_size=None, gain=None, 
                 optimizer='sgd', lr=1e-3, patience=5, interval=1, img_size=(299, 299), transfrom=None, 
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

#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.opt, mode='min', factor=0.2, patience=patience)
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None, convert_to_uint8=False, epoch_size=10):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_epoch * epoch_size)
        torch.cuda.manual_seed(1)
        best_score = 255
        step = 1
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            self.net.train()
            for batch_idx, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                if self.targeted:
                    if len(data) == 2:
                        target = torch.randint_like(label, 0, self.num_classes).cuda()
                    else:
                        target = data[2].cuda()
                else:
                    target = label

                self.reset_grad()
                mask, noise = self.net(img, label, target)
                img_mean = img.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                img_masked = img * mask + img_mean * (1 - mask)
                if self.targeted:
                    noise = noise * mask
                noise_img = img_masked + noise
                
                if convert_to_uint8:
                    noise_img_uint8 = converter.float32_to_uint8(noise_img)
                    noise_img = converter.uint8_to_float32(noise_img_uint8)
                
                cls = [i(noise_img) for i in self.classifier]
                
                loss_cls, loss_noise = self.get_loss(cls, target if self.targeted else label, noise_img, img)
                
                loss = sum(loss_cls) / len(loss_cls) * 64# + loss_noise
                
                loss.backward()
                self.opt.step()
                scheduler.step(step)
                if writer:
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=step)
                    for i, l in enumerate(loss_cls):
                        writer.add_scalar(
                            'loss_cls_{}'.format(i), l.data, global_step=step)
                    writer.add_scalar(
                        'loss_noise', loss_noise.data, global_step=step)
                    writer.add_scalar(
                        'loss', loss.data, global_step=step)
                step += 1
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                acc, perturbation, imgs, noise_imgs = self.test()
                if writer:
                    for i, (a, p) in enumerate(zip(acc, perturbation)):
                        writer.add_scalar(
                            'acc_{}'.format(i), a, global_step=epoch)
                        writer.add_scalar(
                            'perturbation_{}'.format(i), p, global_step=epoch)
                        
                    acc_mean = sum(acc) / len(acc)
                    perturbation_mean = sum(perturbation) / len(perturbation)
                    if self.targeted:
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
                    noise_imgs = vutils.make_grid(noise_imgs, normalize=True, scale_each=True)
                    writer.add_image('Noise Imgs', noise_imgs, epoch)
                    
#                 self.scheduler.step(score)
                if best_score > score:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            pred = [[]] * len(self.classifier)
            gt = []
            perturbations = []
            imgs = []
            noise_imgs = []
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                if self.targeted:
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
                img_masked = img * mask + img_mean * (1 - mask)
                noise_img = img_masked + noise
                
                noise_img_uint8 = converter.float32_to_uint8(noise_img)
                noise_img = converter.uint8_to_float32(noise_img_uint8)
                
                if batch_idx < 4:
                    imgs.append(img.cpu())
                    noise_imgs.append(noise_img.cpu())
                
                perturbation = noise_img_uint8 - converter.float32_to_uint8(img)
                perturbation = F.interpolate(perturbation, self.img_size, mode='bilinear', align_corners=True)
                perturbation = torch.sqrt((perturbation.view(perturbation.size(0), -1) ** 2).mean(dim=1))
                perturbations.append(perturbation.detach().cpu())
                
                cls = [i(noise_img) for i in self.classifier]
                
#                 print([F.softmax(i, dim=1).max(dim=1) for i in cls])
#                 print(label)
                
                for i, c in enumerate(cls):
                    pred[i].append(c.argmax(dim=1).detach().cpu())
                
            pred = [torch.cat(i).numpy() for i in pred]
            gt = torch.cat(gt).numpy()
            perturbations = torch.cat(perturbations).numpy()
            imgs = torch.cat(imgs)
            noise_imgs = torch.cat(noise_imgs)
            acc = [metrics.accuracy_score(gt, i) for i in pred]
            
            if self.targeted:
                perturbation = [perturbations[gt == p].mean() if (gt == p).sum() > 0 else 0 for p in pred]
            else:
                perturbation = [perturbations[gt != p].mean() if (gt != p).sum() > 0 else 0 for p in pred]
        return acc, perturbation, imgs, noise_imgs

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single, '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single, '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path).state_dict())
    
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
    
    def get_loss(self, pred, target, noise_img, img):
        if self.targeted:
#             target_one_hot = torch.zeros_like(pred[0])
#             target_one_hot.scatter_(1, target.view(-1, 1), 1)
#             loss_cls = []
#             for p in pred:
#                 prob = F.softmax(p, dim=1)
#                 prob_gt = prob.gather(1, target.view(-1, 1))
#                 prob_max = prob.clone()
#                 prob_max[target_one_hot.type(torch.uint8)] = 0
#                 prob_max = prob_max.max(dim=1)[0]
#                 prob_delta = prob_gt - prob_max
#                 loss_cls.append(1 - torch.sigmoid(10 * prob_delta).mean())
            loss_cls = [F.cross_entropy(i, target) for i in pred]
        else:
            target_one_hot = torch.zeros_like(pred[0])
            target_one_hot.scatter_(1, target.view(-1, 1), 1)
            loss_cls = []
            for p in pred:
                prob = F.softmax(p, dim=1)
                prob_gt = prob.gather(1, target.view(-1, 1))
                prob_max = prob.clone()
                prob_max[target_one_hot.type(torch.uint8)] = 0
                prob_max = prob_max.max(dim=1)[0]
                prob_delta = prob_max - prob_gt
                loss_cls.append(1 - torch.sigmoid(10 * prob_delta).mean())
#             loss_cls = [(-torch.log(torch.clamp(1 - F.softmax(i, dim=1), min=1e-6)) * 
#                          target_one_hot).sum() / target_one_hot.size(0) 
#                         for i in pred]
#             loss_cls = [(F.softmax(i, dim=1) * target_one_hot).sum() / target_one_hot.size(0) for i in pred]
            
        noise_img_uint8 = (noise_img * 0.5 + 0.5) * 255
#         noise_img_uint8 = torch.clamp(noise_img_uint8, min=0, max=255)
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        img_uint8 = (img * 0.5 + 0.5) * 255
        img_uint8 = F.interpolate(img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        loss_noise = F.mse_loss(noise_img_uint8, img_uint8)# / 64.
        loss_noise = torch.sqrt(loss_noise)
        return loss_cls, loss_noise