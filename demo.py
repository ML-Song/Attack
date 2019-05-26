#coding=utf-8
import os
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
        
    img = [Image.open(i) for i in glob.iglob('data/IJCAI_2019_AAAC_train/00000/*.jpg')][: 16]
    target = [0] * len(img)
    result = []
    epoch = math.ceil(len(img) / batch_size)
    
    solver = Attack(classifiers[: 2], classifiers[: 2], 
                    device='cuda', patience=2, max_iteration=max_iteration)
    for i in tqdm.tqdm(range(epoch), total=epoch):
        torch.cuda.empty_cache()
        batch_x = img[i * batch_size: (i + 1) * batch_size]
        batch_y = target[i * batch_size: (i + 1) * batch_size]
        result.extend(solver.predict(batch_x, batch_y, max_perturbation=max_perturbation, lr=lr))
        