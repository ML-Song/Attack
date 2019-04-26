#coding=utf-8
import imp
import csv
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from sklearn import metrics
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.datasets.folder import *
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import unet
from config import *
from utils.converter import float32_to_uint8, uint8_to_float32
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    comment = 'model: {}, with_target: {}, beta: {}, with_transform: {}'.format('UNet', with_target, beta, with_transform)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    pretrained_models = []
    for name in model_names:
        MainModel = imp.load_source('MainModel', "models/{}.py".format(name))
        pretrained_model = torch.load('models/{}.pth'.format(name)).cpu()
        if len(devices) > 1:
            pretrained_model = nn.DataParallel(pretrained_model, device_ids=range(len(devices))).cuda()
        pretrained_models.append(pretrained_model)

    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)

    model_dimension = 224
    center_crop = 224
    if with_transform:
        data_augmentation = tv.transforms.Compose([
            tv.transforms.RandomRotation(30), 
            tv.transforms.ColorJitter(0.1), 
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

    # just use cpu for example
    for pretrained_model in pretrained_models:
        pretrained_model = pretrained_model.cuda()
        pretrained_model.eval()
        pretrained_model.volatile = True
    
    if with_target:
        attack_net_single = unet.UNet(3, 3 * num_classes, batch_norm=True)
    else:
        attack_net_single = unet.UNet(3, 3, batch_norm=True)
        
    if len(devices) == 1:
        attack_net = attack_net_single.cuda()
    else:
        attack_net = nn.DataParallel(attack_net_single, device_ids=range(len(devices))).cuda()

    train_dataset = image_from_json.ImageDataSet('data/IJCAI_2019_AAAC_train/info.json', transform=train_transform)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, True, num_classes * epoch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=16, batch_size=batch_size, sampler=train_sampler)
    
    test_dataset = image_list_folder.ImageListFolder(root='dev_data/', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=16, batch_size=batch_size, 
                                              shuffle=True, drop_last=False)

    criterion_cls_target = nn.CrossEntropyLoss()
    criterion_cls_non_target = nn.NLLLoss()
    criterion_min_noise = nn.MSELoss()
    optim = torch.optim.SGD(attack_net_single.parameters(), lr=lr, weight_decay=5e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', verbose=True, patience=20, factor=0.2, threshold=5e-3)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with SummaryWriter(comment=comment) as writer:
        step = 0
        best_score = 255.
        for epoch in range(max_epoch):
            attack_net_single.train()
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
                
#                 batch_x_with_noise = float32_to_uint8(batch_x_with_noise)
#                 batch_x_with_noise = uint8_to_float32(batch_x_with_noise)
                outs = [pretrained_model(batch_x_with_noise) for pretrained_model in pretrained_models]
                
                loss_min_noise = criterion_min_noise(batch_x_with_noise, batch_x)
                if with_target:
                    loss_cls_target = sum([criterion_cls_target(out, batch_y) for out in outs]) / len(outs)
                    loss = (loss_cls_target + beta * loss_min_noise) / (1 + beta)
                    writer.add_scalar('loss_cls_target', loss_cls_target.data, global_step=step)
                else:
                    target_one_hot = torch.zeros((n, num_classes), dtype=torch.float32, device=batch_y.device)
                    target_one_hot.scatter_(1, batch_y.view(-1, 1), 1)
                    loss_cls_non_target = sum([(F.softmax(out, dim=1) * target_one_hot).mean() for out in outs]) / len(outs)
#                     out_inverses = [torch.log(torch.clamp(1 - torch.softmax(out, dim=1), min=1e-6, max=1)) for out in outs]
                    
#                     loss_cls_non_target = sum([criterion_cls_non_target(out_inverse, batch_y) \
#                                                for out_inverse in out_inverses]) / len(outs)
                    loss = (loss_cls_non_target + beta * loss_min_noise) / (1 + beta)
                    writer.add_scalar('loss_cls_non_target', loss_cls_non_target.data, global_step=step)
                loss.backward()
                optim.step()
                
                writer.add_scalar('loss_min_noise', loss_min_noise.data, global_step=step)
                writer.add_scalar('loss', loss.data, global_step=step)
                writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=step)
                step += 1
            attack_net_single.eval()
            with torch.no_grad():
                right_nums = [0] * len(model_names)
                scores = [0] * len(model_names)
                predictions = []
                gt = []
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
                    batch_x_with_noise = uint8_to_float32(batch_x_with_noise)
                    
                    outs = [pretrained_model(batch_x_with_noise).detach().cpu() for pretrained_model in pretrained_models]
                    gt.append(batch_y)
                    
                    predictions.append([out.argmax(dim=1).numpy().tolist() for out in outs])
                    
                    if epoch % interval == 0:
                        original_images.append(batch_x.detach().cpu())
                        noise_images.append(batch_x_with_noise.detach().cpu())
                    
                    for j, out in enumerate(outs):
                        if with_target:
                            right_index = out.argmax(dim=1) != batch_y
                            wrong_index = out.argmax(dim=1) == batch_y
                        else:
                            right_index = out.argmax(dim=1) == batch_y
                            wrong_index = out.argmax(dim=1) != batch_y

                        right_nums[j] += right_index.sum()
                        if wrong_index.sum() > 0:
                            scores[j] += torch.sqrt((((batch_x_with_noise - batch_x).detach().cpu()[wrong_index] * 128) ** 2).mean()) * wrong_index.sum()
                        
                gt = torch.cat(gt).numpy()
                predictions = np.concatenate(predictions, axis=-1)
                scores = [(s + r * 128) / len(test_dataset) for s, r in zip(scores, right_nums)]
                
                score_mean = float(sum(scores) / len(scores))
                scheduler.step(score_mean)
                
                accs = [metrics.accuracy_score(p, gt) for p in predictions]
                
                for j, (acc, score) in enumerate(zip(accs, scores)):
                    writer.add_scalar('acc_{}'.format(j), acc, global_step=epoch)
                    writer.add_scalar('score_{}'.format(j), score, global_step=epoch)

                writer.add_scalar('score_mean', score_mean, global_step=epoch)
                if epoch % interval == 0:
                    original_images = torch.cat(original_images)[: 200]
                    noise_images = torch.cat(noise_images)[: 200]

                    original_images = tv.utils.make_grid(
                        original_images, normalize=True, scale_each=True)
                    noise_images = tv.utils.make_grid(
                        noise_images, normalize=True, scale_each=True)
                    writer.add_image('original_images', original_images, global_step=epoch)
                    writer.add_image('noise_images', noise_images, global_step=epoch)
            if best_score > score_mean:
                best_score = score_mean
                torch.save(attack_net_single.state_dict(), '{}/best_{}.pt'.format(checkpoint_dir, comment))