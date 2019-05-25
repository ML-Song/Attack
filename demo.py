#coding=utf-8
import os
import math
import glob
import torch
from PIL import Image

import pretrainedmodels
from config_gba import *
from models.gradient_based_attack import Attack
from models.classifier import ClassifierNet, Classifier


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))
    models = [pretrainedmodels.__dict__[name]() for name in classifier_name]
    classifiers = [ClassifierNet(model, num_classes) for model in models]
    for c, p in zip(classifiers, classifier_path):
        c.load_state_dict(torch.load(p))
        
    solver = Attack(classifiers, device='cuda', patience=2)
    img = [Image.open(i) for i in glob.iglob('data/dev_data/*.png')]
    target = [1] * len(img)
    result = []
    for i in range(math.ceil(len(img) / batch_size)):
        torch.cuda.empty_cache()
        batch_x = img[i * batch_size: (i + 1) * batch_size]
        batch_y = target[i * batch_size: (i + 1) * batch_size]
        result.extend(solver.predict(batch_x, batch_y, True, max_perturbation=10))
        