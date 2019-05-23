#coding=utf-8
import os
import tqdm
import torch
from torch import nn
import torchvision as tv
from sklearn import metrics
from torch.nn import functional as F

from utils import converter


class AttackNet(nn.Module):
    def __init__(self, backbone, out_channels=3):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        
    def forward(self, x, target=None):
        n, c, h, w = x.shape
        noise = self.backbone(x)
        if target is None:
            return noise
        else:
            noise = noise.view(n, -1, self.out_channels, h, w)
            index = target.view(-1, 1, 1, 1, 1)
            index = index.repeat(1, 1, self.out_channels, h, w)
            noise = noise.gather(1, index).squeeze(dim=1)
#             noise = torch.cat([noise[i, batch_y[i]].unsqueeze(0) for i in range(n)], dim=0)
            return noise
        
        
class Attack(object):
    def __init__(self, net, classifier=None, train_loader=None, test_loader=None, batch_size=None, gain=None, 
                 optimizer='sgd', lr=1e-3, patience=5, interval=1, img_size=(299, 299), weight=1, 
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
        self.weight = weight
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        self.net_single = net
        self.classifier_single = classifier
        for i in self.classifier_single:
            i.eval()

        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.net = self.net_single.cuda()
            self.classifier = [i.cuda() for i in self.classifier_single]
        else:
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).cuda()
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
                    target = None

                self.reset_grad()
                noise = self.net(img, target)
                noise_img = img + noise
                
                if convert_to_uint8:
                    noise_img_uint8 = converter.float32_to_uint8(noise_img)
                    noise_img = converter.uint8_to_float32(noise_img_uint8)
                
                cls = [i(noise_img) for i in self.classifier]
                
                loss_cls, loss_noise = self.get_loss(cls, target if self.targeted else label, noise_img, img)
                
                loss = sum(loss_cls) / len(loss_cls) + loss_noise * self.weight
                
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
                acc, perturbation = self.test()
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
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1]
                if self.targeted:
                    if len(data) == 2:
                        target = torch.randint_like(label, 0, self.num_classes).cuda()
                    else:
                        target = data[2].cuda()
                    gt.append(target.cpu())
                else:
                    target = None
                    gt.append(label.cpu())
                    
                noise = self.net(img, target)
                noise_img = img + noise
                
                noise_img_uint8 = converter.float32_to_uint8(noise_img)
                noise_img = converter.uint8_to_float32(noise_img_uint8)
                
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
            
            acc = [metrics.accuracy_score(gt, i) for i in pred]
            
            if self.targeted:
                perturbation = [perturbations[gt == p].mean() if (gt == p).sum() > 0 else 0 for p in pred]
            else:
                perturbation = [perturbations[gt != p].mean() if (gt != p).sum() > 0 else 0 for p in pred]
        return acc, perturbation

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single, '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single, '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path).state_dict())
    
    def predict(self, img, target=None):
        x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        self.net.eval()
        with torch.no_grad():
            noise = self.net(x, target)
            noise_img = x + noise
            noise_img_uint8 = converter.float32_to_uint8(noise_img)
            noise_img_uint8 = noise_img_uint8.detach().cpu().numpy().astype(np.uint8)
            noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
        return noise_img_uint8
    
    def get_loss(self, pred, target, noise_img, img):
        if self.targeted:
            loss_cls = [F.cross_entropy(i, target) for i in pred]
        else:
            target_one_hot = torch.zeros_like(pred[0])
            target_one_hot.scatter_(1, target.view(-1, 1), 1)
            loss_cls = [(-torch.log(torch.clamp(1 - F.softmax(i, dim=1), min=1e-6)) * 
                         target_one_hot).sum() / target_one_hot.size(0) 
                        for i in pred]
#             loss_cls = [(-F.log_softmax(i, dim=1) * target_one_hot).sum() / target_one_hot.size(0) for i in pred]
            
        noise_img_uint8 = noise_img#(noise_img * 0.5 + 0.5) * 255
        noise_img_uint8 = F.interpolate(noise_img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        img_uint8 = img#(img * 0.5 + 0.5) * 255
        img_uint8 = F.interpolate(img_uint8, self.img_size, mode='bilinear', align_corners=True)
        
        loss_noise = F.mse_loss(noise_img_uint8, img_uint8)# / 64.
        return loss_cls, loss_noise