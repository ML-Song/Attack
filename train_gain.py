#coding=utf-8
import os
import math
import torch
from torch import nn
import pretrainedmodels
import torchvision as tv
from tensorboardX import SummaryWriter

from config_gain import *
from models import drn, activations
from models.gain import GAIN, GAINSolver
from dataset import image_from_json, image_list_folder


if __name__ == '__main__':
    checkpoint_name = 'GAIN model: {} optimizer: {} loc: {} temp: {}'.format(model_name, optimizer, loc, temp)
    comment = 'GAIN model: {} optimizer: {} loc: {} temp: {}'.format(model_name, optimizer, loc, temp)
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
    
    if model_name in pretrainedmodels.model_names:
        backbone = pretrainedmodels.__dict__[model_name]()
        backbone = nn.Sequential(*list(backbone.children())[: -2])
    elif model_name == 'drn_d_54':
        backbone = drn.drn_d_54(True, out_feat=True)
    else:
        raise Exception('\nModel {} not exist'.format(model_name))
#     net = GAIN(backbone, num_classes, in_channels=in_channels)
    solver = GAINSolver(backbone, num_classes, in_channels, 
                        train_loader, test_loader, test_batch_size, 
                        lr=lr, loss_weights=loss_weights, checkpoint_name=checkpoint_name, 
                        devices=devices, area_threshold=area_threshold, 
                        optimizer=optimizer, activation=activations.HardConcrete(loc, temp) if loc is not None else None)
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer, epoch_size=math.ceil(num_classes * epoch_size / train_batch_size))