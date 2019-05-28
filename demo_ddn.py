#coding=utf-8
import os
import csv
import math
import tqdm
import time
import torch
import numpy as np
from PIL import Image
import pretrainedmodels
import torchvision as tv


from config_ddn import *
from attack import attack
from models.ddn import DDN
from models.fgm_l2 import FGM_L2
from utils.ddn_utils import NormalizedModel
from models.classifier import ClassifierNet, Classifier


if __name__ == '__main__':
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    image_mean = torch.tensor(image_mean).view(1, 3, 1, 1).cuda()
    image_std = torch.tensor(image_std).view(1, 3, 1, 1).cuda()
    
    white_models = [pretrainedmodels.__dict__[name](pretrained=None) for name in white_model_name]
    white_models = [ClassifierNet(model, num_classes) for model in white_models]
    for m, p in zip(white_models, white_model_path):
        m.load_state_dict(torch.load(p))
        m.eval()
        m = m.cuda()
        m = NormalizedModel(m, image_mean, image_std)
        
    black_model = pretrainedmodels.__dict__[black_model_name](pretrained=None)
    black_model = ClassifierNet(black_model, num_classes)
    black_model.load_state_dict(torch.load(black_model_path))
    black_model.eval()
    black_model = black_model.cuda()
    black_model = NormalizedModel(black_model, image_mean, image_std)
        
    def black_box_model(img):
        t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
        t_img = t_img.unsqueeze(0).to(device)
        with torch.no_grad():
            return black_model(t_img).argmax().detach().cpu().numpy()
    
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
                
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    surrogate_models = white_models
    attacks = [
        DDN(steps, device=device),
        FGM_L2(1)
    ]
    result = []
#     score = 0
    images = np.array([np.asarray(i.resize((224, 224))) for i in data['imgs']])
    max_epoch = math.ceil(len(images) / batch_size)
    correct_num = 0
    for epoch in tqdm.tqdm(range(max_epoch)):
        img_np = images[batch_size * epoch: batch_size * (epoch + 1)]#[0]
        label = data['target'][batch_size * epoch: batch_size * (epoch + 1)]#[0]
        adv = attack(black_box_model, surrogate_models, attacks,
                     img_np, label, targeted=targeted, device=device)
        
        pred_labels = np.array([black_box_model(a) for a in adv])
        correct_num += (pred_labels == label).sum()
#         print(correct_num)
        result.append(adv)
    result = np.concatenate(result)
    print(correct_num)
#     print('score: {}'.format(score / len(data['target'])))
    result = [Image.fromarray(img.astype(np.uint8)) for i, img in enumerate(result)]
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for img, path in zip(result, data['image_path']):
            name = path.split('/')[-1]
            path = os.path.join(output_dir, name)
            img.save(path)
    print(time.time() - start)
    