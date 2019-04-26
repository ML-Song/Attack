#coding=utf-8
import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from sklearn import metrics
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from config_gain import *
from models import gain
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    comment = 'GAIN with_transform: {}'.format(with_transform)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))

    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)
    model_dimension = 224
    center_crop = 224
    if with_transform:
        data_augmentation = tv.transforms.Compose([
            tv.transforms.RandomRotation(30),
#             tv.transforms.ColorJitter(0.1),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip()
        ])
    else:
        data_augmentation = tv.transforms.Compose([])

    train_transform = transforms.Compose([
        data_augmentation,
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = image_from_json.ImageDataSet(
        'data/IJCAI_2019_AAAC_train/info.json', transform=train_transform)
    train_sampler = torch.utils.data.sampler.RandomSampler(
        train_dataset, True, num_classes * epoch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=16, batch_size=batch_size, sampler=train_sampler)

    test_dataset = image_list_folder.ImageListFolder(
        root='dev_data/', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=16, batch_size=batch_size,
                                              shuffle=True, drop_last=False)
    
    gai_net_single = gain.GAIN(num_classes)
    if len(devices) == 1:
        gai_net = gai_net_single.cuda()
    else:
        gai_net = nn.DataParallel(gai_net_single, device_ids=range(len(devices))).cuda()
    criterion = gain.GAINLoss()
    optim = torch.optim.SGD(gai_net_single.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'max', verbose=True, patience=20, factor=0.2, threshold=5e-3)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    with SummaryWriter(comment=comment) as writer:
        step = 0
        best_acc = 0
        for epoch in range(max_epoch):
            gai_net_single.train()
            writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=step)
            for i, batch_data in tqdm.tqdm(enumerate(train_loader)):
                batch_x = batch_data[0].cuda()
                batch_y = batch_data[1].cuda()
                n, c, h, w = batch_x.shape
                optim.zero_grad()
                out, out_masked, cam = gai_net(batch_x)
                loss, (loss_cls, loss_mining, loss_seg) = criterion(out, out_masked, cam, batch_y)
                loss.backward()
                optim.step()
                
                writer.add_scalar('loss_cls', loss_cls.data, global_step=step)
                writer.add_scalar('loss_mining', loss_mining.data, global_step=step)
                writer.add_scalar('loss_seg', loss_seg.data, global_step=step)
                writer.add_scalar('loss', loss.data, global_step=step)
                step += 1
                
            gai_net_single.eval()
            with torch.no_grad():
                predictions = []
                gt = []
                original_images = []
                cams = []
                for i, batch_data in tqdm.tqdm(enumerate(test_loader)):
                    batch_x = batch_data[0].cuda()
                    batch_y = batch_data[1]
                    n, c, h, w = batch_x.shape
                    out, out_masked, cam = gai_net(batch_x)
                    gt.append(batch_y)
                    predictions.append(out.argmax(dim=1).detach().cpu())
                    
                    if epoch % interval == 0:
                        original_images.append(batch_x.detach().cpu())
                        cams.append(cam.detach().cpu())

                gt = torch.cat(gt).numpy()
                predictions = torch.cat(predictions).numpy()
                acc = metrics.accuracy_score(predictions, gt)
                scheduler.step(acc)
                writer.add_scalar('acc', acc, global_step=epoch)
                if epoch % interval == 0:
                    original_images = torch.cat(original_images)[: 200]
                    cams = torch.cat(cams)[: 200]

                    original_images = tv.utils.make_grid(
                        original_images, normalize=True, scale_each=True)
                    cams = tv.utils.make_grid(
                        cams, normalize=True, scale_each=True)
                    writer.add_image('original_images', original_images, global_step=epoch)
                    writer.add_image('cams', cams, global_step=epoch)
            if best_acc < acc:
                best_acc = acc
                torch.save(gai_net_single.state_dict(), '{}/best_{}.pt'.format(checkpoint_dir, comment))