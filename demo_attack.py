#coding=utf-8
import os
import sys
import csv
import time
import math
import torch
import numpy as np
from torch import nn
from PIL import Image
import torchvision as tv

import pretrainedmodels
from config_attack import *
from models.gain import GAIN, GAINSolver
from models.attack import Attack, AttackNet
from models.classifier import ClassifierNet
from models.transformer_net import TransformerNet


if __name__ == '__main__':
    inputs_path = sys.argv[1]
    outputs_path = sys.argv[2]
    start = time.time()
    data = {'image_path': [], 'target': [], 'label': []}
    with open(os.path.join(inputs_path, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(inputs_path, row['filename'])
            data['image_path'].append(img_path)
            data['target'].append(int(row['targetedLabel']))
            data['label'].append(int(row['trueLabel']))
            
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

        gain = GAINSolver(backbone, num_classes, in_channels=in_channels, devices=devices)
        gain.load_model(gain_checkpoint_path)
    else:
        gain = None
    
    unet = TransformerNet(num_classes=num_classes * 3)
    attack_net = AttackNet(unet)
    solver = Attack(attack_net, classifier, test_classifier, targeted, 
                    gain=gain, num_classes=num_classes, as_noise=as_noise)
    solver.load_model(checkpoint_path)
    images = [Image.open(i) for i in data['image_path']]
    labels = data['label']
    targets = data['target']
    result = []
    epoch = math.ceil(len(images) / inference_batch_size)
    for i in range(epoch):
        batch_x = images[i * inference_batch_size: (i + 1) * inference_batch_size]
        batch_y = (targets[i * inference_batch_size: (i + 1) * inference_batch_size] if targeted else 
                   labels[i * inference_batch_size: (i + 1) * inference_batch_size])
        result.append(solver.predict(batch_x, batch_y))
    result = np.concatenate(result)
    result = [Image.fromarray(img.astype(np.uint8)) for i, img in enumerate(result)]
    if outputs_path is not None:
        if not os.path.exists(outputs_path):
            os.mkdir(outputs_path)
        for img, path in zip(result, data['image_path']):
            name = path.split('/')[-1]
            path = os.path.join(outputs_path, name)
            img.save(path)
    print(time.time() - start)