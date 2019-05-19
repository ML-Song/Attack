#coding=utf-8
import os
import tqdm
import torch
from torch import nn
import torchvision as tv
from sklearn import metrics
import torchvision.utils as vutils
from torch.nn import functional as F


class GAIN(nn.Module):
    def __init__(self, backbone, num_classes, in_channels=None):
        super().__init__()
        self.backbone = backbone
        if in_channels is None:
            in_channels = GAIN._get_output_shape(backbone)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x, with_gcam=False, target=None):
        n, c, h, w = x.shape
        feat_map = self.backbone(x)
        feat = self.gap(feat_map).view(n, -1)
        prob = self.fc(feat)
        
        if with_gcam:
            weights = self.fc.weight
            if target is None:
                weight = weights[prob.argmax(-1)]
                gcam = (weight.unsqueeze(-1).unsqueeze(-1) * feat_map).sum(dim=1, keepdim=True)
            else:
                weight = weights[target]
                gcam = (weight.unsqueeze(-1).unsqueeze(-1) * feat_map).sum(dim=1, keepdim=True)
            gcam = F.interpolate(gcam, (h, w), mode='bilinear', align_corners=True)
            gcam_min = gcam.min()
            gcam_max = gcam.max()
            scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
            return prob, scaled_gcam
        else:
            return prob
        
    @staticmethod
    def _get_output_shape(model):
        for layer in list(model.modules())[::-1]:
            if isinstance(layer, nn.Conv2d):
                break
        return layer.out_channels
    
    
class GAINSolver(object):
    def __init__(self, net, train_loader=None, test_loader=None, batch_size=None, 
                 loss_weights=None, optimizer='adam', lr=1e-3, patience=5, interval=1, 
                 checkpoint_dir='saved_models', checkpoint_name='', devices=[0], 
                 transfrom=None, area_threshold=0.2):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.devices = devices
        self.area_threshold = area_threshold
        if transfrom is None:
            self.transfrom = tv.transforms.Compose([
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transfrom = transfrom
            
        if loss_weights is None:
            self.loss_weights = (1, 1)
        else:
            self.loss_weights = loss_weights
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        self.net_single = net
        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.net = self.net_single.cuda()
        else:
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).cuda()
            
        if optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                self.net_single.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        elif optimizer == 'adam':
            self.opt = torch.optim.Adam(
                self.net_single.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise Exception('Optimizer {} Not Exists'.format(optimizer))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max', factor=0.2, patience=patience)
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None, pretrain_percentage=0.0, mode='PRETRAIN'):
        assert(mode in ('PRETRAIN', 'TRAIN'))
        pretrain_epochs = int(max_epoch * pretrain_percentage)
        torch.cuda.manual_seed(1)
        best_score = 0
        step = 1
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            if epoch >= pretrain_epochs:
                mode = 'TRAIN'
            self.net.train()
            for batch_idx, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                img_mean = img.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

                self.reset_grad()
                cls, gcam = self.net(img, with_gcam=True, target=label)
                mask = self._soft_mask(gcam)
                img_masked = img * (1 - mask) + img_mean * mask
                cls_masked = self.net(img_masked)

                if len(self.loss_weights) == 2:
                    loss_cls, loss_am = self.get_loss(cls, cls_masked, label)
                    if mode == 'PRETRAIN':
                        loss = loss_cls
                    elif mode == 'TRAIN':
                        loss = loss_cls * self.loss_weights[0] + loss_am * self.loss_weights[1]
                elif len(self.loss_weights) == 3:
                    loss_cls, loss_am, loss_mask = self.get_loss(cls, cls_masked, label, gcam)
                    if mode == 'PRETRAIN':
                        loss = loss_cls
                    elif mode == 'TRAIN':
                        loss = loss_cls * self.loss_weights[0] + loss_am * self.loss_weights[1] + loss_mask * self.loss_weights[2]
                else:
                    raise Exception('Loss Weights: {} Error'.format(self.loss_weights))
                    
                loss.backward()
                self.opt.step()
                if writer:
                    writer.add_scalar(
                        'loss', loss.data, global_step=step)
                    writer.add_scalar(
                        'loss_cls', loss_cls.data, global_step=step)
                    writer.add_scalar(
                        'loss_mining', loss_am.data, global_step=step)
                    if len(self.loss_weights) == 3:
                        writer.add_scalar(
                            'loss_mask', loss_mask.data, global_step=step)
                step += 1
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                acc, acc_masked, imgs, masks = self.test()
                if writer:
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=epoch)
                    writer.add_scalar(
                        'acc', acc, global_step=epoch)
                    writer.add_scalar(
                        'acc_masked', acc_masked, global_step=epoch)
                    score = acc - acc_masked

#                     imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
#                     writer.add_image('Image', imgs, epoch)

#                     masks = vutils.make_grid(masks, normalize=True, scale_each=True)
#                     writer.add_image('Mask', masks, epoch)

                    imgs = torch.clamp(imgs * 0.5 + 0.5, min=0, max=1)
                    image_with_mask = imgs * (masks * 0.8 + 0.2)
                    image_with_mask = vutils.make_grid(image_with_mask, normalize=False, scale_each=True)
                    writer.add_image('Image With Mask', image_with_mask, epoch)
                
                self.scheduler.step(score)
                if best_score < score:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self, max_num=160):
        self.net.eval()
        with torch.no_grad():
            pred = []
            pred_masked = []
            gt = []
            imgs = []
            masks = []
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1]
                cls, gcam = self.net(img, with_gcam=True, target=label)
                mask = self._soft_mask(gcam)
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                img_masked = img * (1 - mask)
                cls_masked = self.net(img_masked)
                
                pred.append(cls.argmax(dim=1).detach().cpu())
                pred_masked.append(cls_masked.argmax(dim=1).detach().cpu())
                gt.append(label)
                if len(imgs) * self.batch_size < max_num:
                    masks.append(mask.detach().cpu())
                    imgs.append(img.detach().cpu())

            pred = torch.cat(pred).numpy()
            pred_masked = torch.cat(pred_masked).numpy()
            gt = torch.cat(gt).numpy()
            masks = torch.cat(masks)
            imgs = torch.cat(imgs)
            acc = metrics.accuracy_score(gt, pred)
            acc_masked = metrics.accuracy_score(gt, pred_masked)
        return acc, acc_masked, imgs[: max_num], masks[: max_num]

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
            if target is None:
                cls, gcam = self.net(x, with_gcam=True)
            else:
                cls, gcam = self.net(x, with_gcam=True, target=target)
            mask = self._soft_mask(gcam).detach().cpu()
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = mask.permute(0, 2, 3, 1).numpy()
        return cls, mask
    
    def _soft_mask(self, x, scale=10, threshold=0.5):
        return torch.sigmoid(scale * (x - threshold))
    
    def get_loss(self, pred, pred_masked, target, mask=None):
        n, c = pred.shape
        loss_cls = F.cross_entropy(pred, target)
        
#         loss_am = F.cross_entropy(pred_masked, target)
        
        target_one_hot = torch.zeros((n, c), dtype=torch.float32, device=target.device)
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        loss_am = (torch.sigmoid(pred_masked) * target_one_hot).sum() / n
#         loss_am = (-torch.log(torch.clamp(1 - F.softmax(pred_masked, dim=1), min=1e-6)) * target_one_hot).sum() / n
        
        if mask is None:
            return loss_cls, loss_am
        else:
            loss_mask = mask.view(n, -1).mean(dim=1) - self.area_threshold
            loss_mask = torch.clamp(loss_mask, min=0)
            loss_mask = loss_mask.mean()
            return loss_cls, loss_am, loss_mask