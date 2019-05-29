#coding=utf-8
import os
import math
import torch
from torch import nn
import torchvision as tv
from tensorboardX import SummaryWriter

import pretrainedmodels
from config_attack import *
from models.gain import GAIN, GAINSolver
from models.attack import Attack, AttackNet
from models.classifier import ClassifierNet
from models.transformer_net import TransformerNet
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    checkpoint_name = 'Attack targeted: {} beta: {} with_gain: {}'.format(targeted, beta, gain_model_name is not None)
    comment = 'Attack targeted: {} beta: {} with_gain: {}'.format(targeted, beta, gain_model_name is not None)
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
    
    model = [pretrainedmodels.__dict__[i](pretrained=None) for i in classifier_name]
    classifier = [ClassifierNet(i, num_classes) for i in model]
    for c, p in zip(classifier, classifier_path):
        c.load_state_dict(torch.load(p))
        
    model = [pretrainedmodels.__dict__[i](pretrained=None) for i in test_classifier_name]
    test_classifier = [ClassifierNet(i, num_classes) for i in model]
    for c, p in zip(test_classifier, test_classifier_path):
        c.load_state_dict(torch.load(p))
    if gain_model_name is not None:
        if gain_model_name in pretrainedmodels.model_names:
            backbone = pretrainedmodels.__dict__[gain_model_name](pretrained=None)
            backbone = nn.Sequential(*list(backbone.children())[: -2])
        else:
            raise Exception('\nModel {} not exist'.format(model_name))

        gain = GAINSolver(backbone, num_classes, in_channels=in_channels)
        gain.load_model(gain_checkpoint_path)
    else:
        gain = None
    
    unet = TransformerNet(num_classes=num_classes * 3)
    attack_net = AttackNet(unet)
    solver = Attack(attack_net, classifier, test_classifier, targeted, 
                    train_loader, test_loader, test_batch_size, gain=gain, beta=beta, optimizer=optimizer, 
                    num_classes=num_classes, lr=lr, checkpoint_name=checkpoint_name, devices=devices)
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer, epoch_size=math.ceil(num_classes * epoch_size / train_batch_size))
        