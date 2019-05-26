#coding=utf-8
import os
import csv
import math
import glob
import tqdm
import torch
from PIL import Image

import pretrainedmodels
from config_gba import *
from models.gradient_based_attack import Attack
from models.classifier import ClassifierNet, Classifier


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    models = [pretrainedmodels.__dict__[name]() for name in classifier_name]
    classifiers = [ClassifierNet(model, num_classes) for model in models]
    for c, p in zip(classifiers, classifier_path):
        c.load_state_dict(torch.load(p))
        
        
    data = {'image_path': [], 'target': [], 'imgs': []}
    with open(os.path.join(data_path, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(data_path, row['filename'])
            data['image_path'].append(img_path)
            data['imgs'].append(Image.open(img_path))
            if targeted:
                data['target'].append(int(row['targetedLabel']))
            else:
                data['target'].append(int(row['trueLabel']))
    result = []
    epoch = math.ceil(len(data['imgs']) / batch_size)
    
    solver = Attack(classifiers[: 2], classifiers[: 2], 
                    device='cuda', patience=patience, max_iteration=max_iteration)
    for i in tqdm.tqdm(range(epoch), total=epoch):
        torch.cuda.empty_cache()
        batch_x = data['imgs'][i * batch_size: (i + 1) * batch_size]
        batch_y = data['target'][i * batch_size: (i + 1) * batch_size]
        result.extend(solver.predict(batch_x, batch_y, targeted, max_perturbation=max_perturbation, lr=lr))
        