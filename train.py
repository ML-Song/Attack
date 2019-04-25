#coding=utf-8
import imp
import csv
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from sklearn import metrics
from tensorboardX import SummaryWriter
from torchvision.datasets.folder import *
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import unet
from config import *
from utils.float32_to_uint8 import float32_to_uint8
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    checkpoint_dir = 'saved_models'
    comment = 'model: {}, with_target: {}, beta: {}'.format('UNet', with_target, beta)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    MainModel = imp.load_source('MainModel', "models/tf_to_pytorch_inception_v1.py")
    pretrained_model = torch.load('models/tf_to_pytorch_inception_v1.pth')

    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)

    model_dimension = 224
    center_crop = 224
    train_transform = transforms.Compose([
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

    # just use cpu for example
    pretrained_model = pretrained_model.cuda()
    pretrained_model.eval()
    pretrained_model.volatile = True
    if with_target:
        attack_net = unet.UNet(3, 3 * num_classes, batch_norm=True).cuda()
    else:
        attack_net = unet.UNet(3, 3, batch_norm=True).cuda()

    train_dataset = image_from_json.ImageDataSet('data/IJCAI_2019_AAAC_train/info.json', transform=train_transform)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, True, num_classes * 5)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=16, batch_size=batch_size, sampler=train_sampler)
    
    test_dataset = image_list_folder.ImageListFolder(root='dev_data/', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=16, batch_size=batch_size, 
                                              shuffle=True, drop_last=False)

    criterion_cls_target = nn.CrossEntropyLoss()
    criterion_cls_non_target = nn.NLLLoss()
    criterion_min_noise = nn.MSELoss()
    optim = torch.optim.SGD(attack_net.parameters(), lr=lr, weight_decay=5e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', verbose=True, patience=20, factor=0.2, threshold=5e-3)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with SummaryWriter(comment=comment) as writer:
        step = 0
        best_acc = 1
        for epoch in range(max_epoch):
            attack_net.train()
            for i, batch_data in tqdm.tqdm(enumerate(train_loader)):
                batch_x = batch_data[0].cuda()
                n, c, h, w = batch_x.shape
                
                optim.zero_grad()
                noise = attack_net(batch_x)
                if with_target:
                    if len(batch_data) == 3: 
                        batch_y = batch_data[2].cuda()
                    else:
                        batch_y = torch.randint(0, num_classes, (batch_x.size(0), ), dtype=torch.int64).cuda()
                    noise = noise.view(n, -1, c, h, w)
                    noise = torch.cat([noise[i, batch_y[i]].unsqueeze(0) for i in range(n)], dim=0)
                else:
                    batch_y = batch_data[1].cuda()

                batch_x_with_noise = batch_x + noise
                batch_x_with_noise = float32_to_uint8(batch_x_with_noise)
                out = pretrained_model(batch_x_with_noise)
                
                loss_min_noise = criterion_min_noise(batch_x_with_noise, batch_x)
                if with_target:
                    loss_cls_target = criterion_cls_target(out, batch_y)
                    loss = (loss_cls_target + beta * loss_min_noise) / (1 + beta)
                    writer.add_scalar('loss_cls_target', loss_cls_target.data, global_step=step)
                else:
                    out_inverse = torch.log(1 - torch.softmax(out, dim=1))
                    loss_cls_non_target = criterion_cls_non_target(out_inverse, batch_y)
                    loss = (loss_cls_non_target + beta * loss_min_noise) / (1 + beta)
                    writer.add_scalar('loss_cls_non_target', loss_cls_non_target.data, global_step=step)
                loss.backward()
                optim.step()
                
                writer.add_scalar('loss_min_noise', loss_min_noise.data, global_step=step)
                writer.add_scalar('loss', loss.data, global_step=step)
                writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=step)
                step += 1
            attack_net.eval()
            with torch.no_grad():
                right_num = 0
                score = 0
                gt = []
                predictions = []
                original_images = []
                noise_images = []
                for i, batch_data in tqdm.tqdm(enumerate(test_loader)):
                    batch_x = batch_data[0].cuda()
                    n, c, h, w = batch_x.shape
                    noise = attack_net(batch_x)
                    if with_target:
                        if len(batch_data) == 3: 
                            batch_y = batch_data[2]
                        else:
                            batch_y = torch.randint(0, num_classes, (batch_x.size(0), ), dtype=torch.int64)
                        noise = noise.view(n, -1, c, h, w)
                        noise = torch.cat([noise[i, batch_y[i]].unsqueeze(0) for i in range(n)], dim=0)
                    else:
                        batch_y = batch_data[1]

                    batch_x_with_noise = batch_x + noise
                    batch_x_with_noise = float32_to_uint8(batch_x_with_noise)
                    out = pretrained_model(batch_x_with_noise).detach().cpu()
                    gt.append(batch_y)
                    predictions.append(out.argmax(dim=1))
                    original_images.append(batch_x.detach().cpu())
                    noise_images.append(batch_x_with_noise.detach().cpu())
                    if with_target:
                        right_index = out.argmax(dim=1) != batch_y
                        wrong_index = out.argmax(dim=1) == batch_y
                    else:
                        right_index = out.argmax(dim=1) == batch_y
                        wrong_index = out.argmax(dim=1) != batch_y
                        
                    right_num += right_index.sum()
                    if wrong_index.sum() > 0:
                        score += torch.sqrt((((batch_x_with_noise - batch_x).detach().cpu()[wrong_index] * 128) ** 2).mean()) * wrong_index.sum()
                gt = torch.cat(gt).numpy()
                predictions = torch.cat(predictions).numpy()
                original_images = torch.cat(original_images)[: 200]
                noise_images = torch.cat(noise_images)[: 200]
                
                original_images = tv.utils.make_grid(
                    original_images, normalize=True, scale_each=True)
                noise_images = tv.utils.make_grid(
                    noise_images, normalize=True, scale_each=True)
                
                score += right_num * 128
                score /= len(test_dataset)
                scheduler.step(float(score))
                acc = metrics.accuracy_score(predictions, gt)
                writer.add_scalar('acc', acc, global_step=epoch)
                writer.add_scalar('score', score, global_step=epoch)
                if epoch % interval == 0:
                    writer.add_image('original_images', original_images, global_step=epoch)
                    writer.add_image('noise_images', noise_images, global_step=epoch)
            if best_acc > acc:
                best_acc = acc
                torch.save(attack_net.state_dict(), '{}/best_{}.pt'.format(checkpoint_dir, comment))