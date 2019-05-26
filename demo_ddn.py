#coding=utf-8
import os
import csv
import math
import glob
import tqdm
import torch
import numpy as np
from PIL import Image
import torchvision as tv

from models import ddn
import pretrainedmodels
from config_ddn import *
from models.classifier import ClassifierNet, Classifier


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    models = [pretrainedmodels.__dict__[name]() for name in classifier_name]
    classifiers = [ClassifierNet(model, num_classes) for model in models]
    for c, p in zip(classifiers, classifier_path):
        c.load_state_dict(torch.load(p))
        c.eval()
        c = c.cuda()
        
        
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
                
    transfrom = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
    #     tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    attacker = ddn.DDN(steps=steps, device=torch.device('cuda: 0'))
    epoch = math.ceil(len(data['imgs']) / batch_size)
    result = []
    for i in tqdm.tqdm(range(epoch), total=epoch):
        torch.cuda.empty_cache()
        batch_x = data['imgs'][i * batch_size: (i + 1) * batch_size]
        batch_y = data['target'][i * batch_size: (i + 1) * batch_size]
        
        batch_x = torch.cat([transfrom(i).unsqueeze(dim=0) for i in batch_x], dim=0).cuda()
        batch_y = torch.tensor(batch_y).cuda()
        
        result.append(attacker.attack(classifiers, batch_x, labels=batch_y, targeted=targeted).detach().cpu().data)
    result = torch.cat(result).numpy()
    result = np.transpose(result, (0, 2, 3, 1))
    result = [Image.fromarray(i) for i in (result * 255).astype(np.uint8)]