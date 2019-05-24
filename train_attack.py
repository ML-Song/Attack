#coding=utf-8
import os
import math
import torch
import torchvision as tv
from tensorboardX import SummaryWriter

import pretrainedmodels
from config_attack import *
from models.unet import UNet
from models.attack import Attack, AttackNet
from models.classifier import ClassifierNet
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    checkpoint_name = 'Attack targeted: {}'.format(targeted, weight)
    comment = 'Attack targeted: {}'.format(targeted, weight)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))

    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = tv.transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)
    
    train_transform = tv.transforms.Compose([
        tv.transforms.Resize(image_size),
        tv.transforms.ToTensor(),
        normalize, 
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize(image_size),
        tv.transforms.ToTensor(),
        normalize, 
    ])

        
    train_dataset = image_from_json.ImageDataSet(
        'data/IJCAI_2019_AAAC_train/train_info.json', transform=train_transform)
    train_sampler = torch.utils.data.sampler.RandomSampler(
        train_dataset, True, num_classes * epoch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=16, batch_size=train_batch_size, sampler=train_sampler)

    vali_dataset = image_from_json.ImageDataSet(
        'data/IJCAI_2019_AAAC_train/vali_info.json', transform=test_transform)
    vali_sampler = torch.utils.data.sampler.RandomSampler(
        vali_dataset, True, num_classes * epoch_size)
    vali_loader = torch.utils.data.DataLoader(
        vali_dataset, num_workers=16, batch_size=test_batch_size, sampler=vali_sampler)
    
    test_dataset = image_list_folder.ImageListFolder(
        root='data/dev_data/', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=16, batch_size=test_batch_size,
                                              shuffle=True, drop_last=False)
    
    model = [pretrainedmodels.__dict__[i]() for i in classifier_name]
    classifier = [ClassifierNet(i, num_classes) for i in model]
    for c, p in zip(classifier, classifier_path):
        c.load_state_dict(torch.load(p).state_dict())

    unet = UNet(n_classes=2 * num_classes * 3)
    attack_net = AttackNet(unet)
    solver = Attack(attack_net, classifier, train_loader, test_loader, test_batch_size, num_classes=num_classes, 
                        lr=lr, checkpoint_name=checkpoint_name, devices=devices, optimizer=optimizer, targeted=targeted)
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer, epoch_size=math.ceil(num_classes * epoch_size / train_batch_size))