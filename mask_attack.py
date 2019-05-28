#coding=utf-8
import os
import csv
import time
import torch
import numpy as np
from torch import nn
from PIL import Image
import pretrainedmodels
import torchvision as tv

from config_mask import *
from models.gain import GAIN, GAINSolver
from models.classifier import Classifier, ClassifierNet

    
def non_targeted_mask_attack(solver, original_image, original_label, test_model=None):
    '''
    image: PIL
    label: int
    '''
    _, original_mask = solver.predict([original_image], [original_label], out_size=(299, 299))
    original_mask[original_mask >= 0.5] = 1
    original_mask[original_mask < 0.5] = 0
    original_mask = original_mask[0]
    
    img_np = np.asarray(original_image)
    img_mean = (img_np * original_mask).sum(axis=0, keepdims=True).sum(axis=1, keepdims=True) / original_mask.sum()
    img_masked = img_np * (1 - original_mask) + img_mean * original_mask
    img_masked = np.round(img_masked)
    img_masked = Image.fromarray((img_masked).astype(np.uint8))
    if test_model is None:
        cls, _ = solver.predict([img_masked], [original_label])
    else:
        cls = test_model.predict([img_masked])
    if cls.argmax() == original_label:
        img_masked = non_targeted_mask_attack(solver, img_masked, original_label)
    return img_masked

    
def targeted_mask_attack(solver, original_image, target_image,
                         original_label, target_label, test_model=None, max_iter=10):
    '''
    image: PIL
    label: int
    '''
    result = original_image

    _, target_mask = solver.predict(
        [target_image], [target_label], out_size=(299, 299))
    target_mask[target_mask >= 0.5] = 1
    target_mask[target_mask < 0.5] = 0
    target_mask = target_mask[0]
    for step in range(max_iter):
        if test_model is None:
            cls, _ = solver.predict([original_image])
        else:
            cls = test_model.predict([original_image])

        cls = cls.argmax()
        if cls == target_label:
            result = original_image
            break
        else:
            if step == 0:
                _, original_mask = solver.predict(
                    [original_image], [original_label], out_size=(299, 299))
            else:
                _, original_mask = solver.predict(
                    [original_image], out_size=(299, 299))
            original_mask[original_mask >= 0.5] = 1
            original_mask[original_mask < 0.5] = 0
            original_mask = original_mask[0]

            img_np1 = np.asarray(original_image)
            img_np2 = np.asarray(target_image)
            img_mean = (img_np1 * original_mask).sum(axis=0,
                                                     keepdims=True).sum(axis=1, keepdims=True) / original_mask.sum()
            img_masked = img_np1 * (1 - original_mask) + \
                img_mean * original_mask
            img_masked = img_masked * (1 - target_mask) + img_np2 * target_mask
            img_masked = np.round(img_masked)
            original_image = Image.fromarray((img_masked).astype(np.uint8))
    return result

# def targeted_mask_attack(solver, original_image, target_image, original_label, target_label, test_model=None):
#     '''
#     image: PIL
#     label: int
#     '''
#     _, target_mask = solver.predict([target_image], [target_label], out_size=(299, 299))
#     target_mask[target_mask >= 0.5] = 1
#     target_mask[target_mask < 0.5] = 0
#     target_mask = target_mask[0]
    
#     _, original_mask = solver.predict([original_image], [original_label], out_size=(299, 299))
#     original_mask[original_mask >= 0.5] = 1
#     original_mask[original_mask < 0.5] = 0
#     original_mask = original_mask[0]
    
#     img_np1 = np.asarray(original_image)
#     img_np2 = np.asarray(target_image)
#     img_mean = (img_np1 * original_mask).sum(axis=0, keepdims=True).sum(axis=1, keepdims=True) / original_mask.sum()
#     img_masked = img_np1 * (1 - original_mask) + img_mean * original_mask
#     img_masked = img_masked * (1 - target_mask) + img_np2 * target_mask
#     img_masked = np.round(img_masked)
#     img_masked = Image.fromarray((img_masked).astype(np.uint8))
#     return img_masked
# #     cls, _ = solver(img_masked)
# #     if cls.argmax() != target_label:
# #         img_masked = targeted_mask_attack(solver, img_masked, target_image, original_label, target_label)
# #     else:
# #         print(cls.argmax(), target_label)
# #         return img_masked



if __name__ == '__main__':
    start = time.time()
    data = {'image_path': [], 'target': [], 'label': []}
    with open(os.path.join(inputs_path, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(inputs_path, row['filename'])
            data['image_path'].append(img_path)
            data['target'].append(int(row['targetedLabel']))
            data['label'].append(int(row['trueLabel']))
            
    if model_name in pretrainedmodels.model_names:
        backbone = pretrainedmodels.__dict__[model_name](pretrained=None)
        backbone = nn.Sequential(*list(backbone.children())[: -2])
    else:
        raise Exception('\nModel {} not exist'.format(model_name))
        
    solver = GAINSolver(backbone, num_classes, in_channels=in_channels)
    solver.load_model(checkpoint_path)
    
    black_model = pretrainedmodels.__dict__[black_model_name](pretrained=None)
    black_model = ClassifierNet(black_model, num_classes)
    black_model.load_state_dict(torch.load(black_model_path))
    black_model.eval()
    black_model = black_model.cuda()
    black_model = Classifier(black_model)
    
    images = np.array([np.asarray(Image.open(i).resize(image_size)) for i in data['image_path']])
    labels = np.array(data['label'])
    targets = np.array(data['target'])
    result = []
    for img, label, target in zip(images, labels, targets):
        original_image = img
        target_image = images[labels == target]
        if targeted:
            result.append(targeted_mask_attack(solver, 
                                               Image.fromarray(original_image), 
                                               Image.fromarray(target_image[0]), label, target, black_model))
        else:
            result.append(non_targeted_mask_attack(solver, Image.fromarray(original_image), label, black_model))
            
    if outputs_path is not None:
        if not os.path.exists(outputs_path):
            os.mkdir(outputs_path)
        for img, path in zip(result, data['image_path']):
            name = path.split('/')[-1]
            path = os.path.join(outputs_path, name)
            img.save(path)
    if with_test:
        cls = black_model.predict(result)
        cls = cls.argmax(axis=1)
        is_success = (targets == cls if targeted else targets != cls)
        result_np = np.array([np.asarray(i.resize(image_size)) for i in result])
        perturbation = np.sqrt(((result_np - images) ** 2).sum(-1)).mean(-1).mean(-1)
        perturbation[~is_success] = 0
        score = (1 - is_success.mean()) * 64 + perturbation.mean()
        print('success: {} perturbation: {} score: {}'.format(is_success.mean(), perturbation.mean(), score))
        
    print(time.time() - start)