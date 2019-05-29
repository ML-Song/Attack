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
    def __init__(self, backbone, out_channels=3, max_perturbation=128.):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.max_perturbation = max_perturbation
        
    def forward(self, x, target):
        n, c, h, w = x.shape
        out = self.backbone(x)
        perturbation = out.view(n, -1, self.out_channels, h, w)
        
        target_index = target.view(-1, 1, 1, 1, 1)
        target_index = target_index.repeat(1, 1, self.out_channels, h, w)
        
        perturbation = perturbation.gather(1, target_index).squeeze(dim=1)
#         perturbation = perturbation / ((128 / self.max_perturbation) * torch.sqrt((perturbation ** 2).sum(dim=1, keepdim=True)))
        perturbation_min = perturbation.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        perturbation_max = perturbation.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        perturbation = ((perturbation - perturbation_min) / (perturbation_max - perturbation_min) - 0.5) * 2
        perturbation = perturbation * (self.max_perturbation / 128)
        return perturbation
        
        
class Attack(object):
    def __init__(self, net, classifier=None, test_classifier=None, targeted=True, 
                 train_loader=None, test_loader=None, batch_size=None, gain=None, 
                 optimizer='sgd', lr=1e-3, patience=5, interval=1, weight=64, beta=8, 
                 img_size=(299, 299), transfrom=None, loss_mode='cross_entropy', margin=0.5, 
                 checkpoint_dir='saved_models', checkpoint_name='', devices=[0], num_classes=110):
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
        self.beta = beta
        
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
        self.test_classifier_single = test_classifier
        if isinstance(classifier, list):
            for i in self.classifier_single:
                i.eval()
                i.volatile = True
                
        if isinstance(test_classifier, list):
            for i in self.test_classifier_single:
                i.eval()
                i.volatile = True

        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.net = self.net_single.cuda()
            if isinstance(classifier, list):
                self.classifier = [i.cuda() for i in self.classifier_single]
            if isinstance(test_classifier, list):
                self.test_classifier = [i.cuda() for i in self.test_classifier_single]
        else:
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).cuda()
            if isinstance(classifier, list):
                self.classifier = [nn.DataParallel(i, device_ids=range(len(devices))).cuda() for i in self.classifier_single]
            if isinstance(test_classifier, list):
                self.test_classifier = [nn.DataParallel(i, device_ids=range(len(devices))).cuda() 
                                        for i in self.test_classifier_single]
        
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
        
    def train(self, max_epoch, writer=None, epoch_size=10):
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
                    target = None

                self.reset_grad()
                perturbation = self.net(img, target if self.targeted else label)
                perturbated_img = perturbation#img + perturbation
                
                cls = [c(perturbated_img) for c in self.classifier]
                
                loss_cls, loss_perturbation = self.get_loss(cls, target if self.targeted else label, 
                                                            perturbated_img, img, self.targeted)
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
                success_rate, perturbation, imgs, perturbated_imgs = self.test(self.targeted)
                if writer:
                    for i, (a, p) in enumerate(zip(success_rate, perturbation)):
                        writer.add_scalar(
                            'success_rate_{}'.format(i), a, global_step=epoch)
                        writer.add_scalar(
                            'perturbation_{}'.format(i), p, global_step=epoch)
                        
                    success_rate_mean = sum(success_rate) / len(success_rate)
                    perturbation_mean = sum(perturbation) / len(perturbation)
                    score = (1 - success_rate_mean) * 64 + perturbation_mean
                    
                    writer.add_scalar(
                            'success_rate_mean', success_rate_mean, global_step=epoch)
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

    def test(self, targeted=False):
        self.net.eval()
        with torch.no_grad():
            preds = []
            gt = []
            perturbations = []
            imgs = []
            perturbated_imgs = []
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                if targeted:
                    if len(data) == 2:
                        target = torch.randint_like(label, 0, self.num_classes).cuda()
                    else:
                        target = data[2].cuda()
                    gt.append(target.cpu())
                else:
                    target = label.cuda()
                    gt.append(label.cpu())
                    
                perturbation = self.net(img, target if targeted else label)
                perturbated_img = perturbation#img + perturbation

                perturbated_img_uint8 = converter.float32_to_uint8(perturbated_img)
                perturbated_img = converter.uint8_to_float32(perturbated_img_uint8)
                
                if batch_idx < 4:
                    imgs.append(img.cpu())
                    perturbated_imgs.append(perturbated_img.cpu())
                
                img_uint8 = converter.float32_to_uint8(img)

                perturbation = perturbated_img_uint8 - img_uint8
                perturbation = F.interpolate(perturbation, self.img_size, mode='bilinear', align_corners=True)
                perturbation = torch.sqrt((perturbation ** 2).sum(dim=1)).mean(dim=-1).mean(dim=-1)
                perturbations.append(perturbation.detach().cpu())
                
                cls = [c(perturbated_img) for c in self.test_classifier]
                preds.append([c.argmax(dim=1).detach().cpu() for c in cls])
                
            preds = [torch.cat([p[i] for p in preds]).numpy() for i, c in enumerate(self.test_classifier)]
            gt = torch.cat(gt).numpy()
            perturbations = torch.cat(perturbations).numpy()
            imgs = torch.cat(imgs)
            perturbated_imgs = torch.cat(perturbated_imgs)
            success_rate = [(gt == p if targeted else gt != p).astype(np.float32).mean() for p in preds]
            
            if targeted:
                perturbation = [perturbations[gt == p].mean() if (gt == p).sum() > 0 else 0 for p in preds]
            else:
                perturbation = [perturbations[gt != p].mean() if (gt != p).sum() > 0 else 0 for p in preds]
        return success_rate, perturbation, imgs, perturbated_imgs

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single.state_dict(), '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single.state_dict(), '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path))
    
    def predict(self, img, label):
        x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        label_tensor = torch.tensor(label).cuda()
        self.net.eval()
        with torch.no_grad():
            perturbation = self.net(x, label)
            perturbated_img = img + perturbation
            perturbated_img_uint8 = converter.float32_to_uint8(perturbated_img)
            perturbated_img_uint8 = perturbated_img_uint8.detach()
            perturbated_img_uint8 = F.interpolate(perturbated_img_uint8, self.img_size, mode='bilinear', align_corners=True)
            perturbated_img_uint8 = perturbated_img_uint8.cpu().numpy().astype(np.uint8)
            perturbated_img_uint8 = np.transpose(perturbated_img_uint8, (0, 2, 3, 1))
        return perturbated_img_uint8
    
    def get_loss(self, preds, target, perturbated_img, img, targeted, max_perturbation=10):
        loss_perturbation = F.mse_loss(perturbated_img, img)
        if targeted:
            loss_cls = 0
            for out in preds:
                loss_cls = loss_cls + F.cross_entropy(out, target)
        else:
            loss_cls = 0
            for out in preds:
                out_inverse = torch.log(torch.clamp(1 - torch.softmax(out, dim=1), min=1e-6))
                loss_cls = loss_cls + F.nll_loss(out_inverse, target)
        return loss_cls, self.beta * loss_perturbation
        
        
        
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