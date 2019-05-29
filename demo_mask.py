#coding=utf-8
import os
import sys
import csv
import time
import torch
import skimage
import numpy as np
from torch import nn
from PIL import Image
import pretrainedmodels
import torchvision as tv
from skimage import feature
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.segmentation import slic

from config_mask import *
from models.gain import GAIN, GAINSolver
from models.classifier import Classifier, ClassifierNet

    
def non_targeted_mask_attack(solver, original_image, original_label, test_model=None, max_iter=10):
    '''
    image: PIL
    label: int
    '''
    if max_iter == 0:
        return original_image
    _, original_mask = solver.predict([original_image], [original_label], out_size=(299, 299))
    original_mask[original_mask >= 0.5] = 1
    original_mask[original_mask < 0.5] = 0
    original_mask = original_mask[0]
    
    img_np = np.asarray(original_image)
#     img_mean = (img_np * original_mask).sum(axis=0, keepdims=True).sum(axis=1, keepdims=True) / original_mask.sum()
#     masked_img = skimage.filters.gaussian(img_np, sigma=0.5, multichannel=True) * 255
#     masked_img = img_np * original_mask
    masked_img = img_np * original_mask
    edges = feature.canny(rgb2gray(masked_img), sigma=1)
    edges = edges.reshape(299, 299, 1).repeat(3, -1)
    edges_mean = masked_img.mean()
    masked_img[edges] = edges_mean
    masked_img = masked_img * original_mask
#     segments = slic(masked_img, n_segments=600, compactness=10)
#     for i in np.unique(segments):
#         for c in range(3):
#             masked_img_c = masked_img[:, :, c]
#             masked_img_c[segments == i] = masked_img_c[segments == i].mean()

    img_masked = img_np * (1 - original_mask) + masked_img * original_mask
    img_masked = np.round(img_masked)
    img_masked = Image.fromarray((img_masked).astype(np.uint8))
    if test_model is None:
        cls, _ = solver.predict([img_masked], [original_label])
    else:
        cls = test_model.predict([img_masked])
    if cls.argmax() == original_label:
        img_masked = non_targeted_mask_attack(solver, img_masked, original_label, test_model, max_iter-1)
    return img_masked

    
def targeted_mask_attack(solver, original_image, target_image,
                         original_label, target_label, test_model=None, max_iter=10):
    '''
    image: PIL
    label: int
    '''
    result = None

    _, target_mask = solver.predict(
        [target_image], [target_label], out_size=(299, 299))
    target_mask[target_mask >= 0.5] = 0.6
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
            img_np1 = np.asarray(original_image)
            img_np2 = np.asarray(target_image)
            if step == 0:
                _, original_mask = solver.predict(
                    [original_image], [original_label], out_size=(299, 299))
            else:
                _, original_mask = solver.predict(
                    [original_image], out_size=(299, 299))
            original_mask[original_mask >= 0.5] = 0.4
            original_mask[original_mask < 0.5] = 0
            original_mask = original_mask[0]

            if step == -1:
                img_mean = (img_np1 * original_mask).sum(axis=0, 
                                                         keepdims=True).sum(axis=1, keepdims=True) / original_mask.sum()
                img_masked = img_np1 * (1 - original_mask) + \
                    img_mean * original_mask
                img_masked = img_masked * (1 - target_mask) + img_np2 * target_mask
            else:
                img_masked = img_np1 * (1 - original_mask) + img_np2 * original_mask
                img_masked = img_masked * (1 - target_mask) + img_np2 * target_mask
            img_masked = np.round(img_masked)
            original_image = Image.fromarray((img_masked).astype(np.uint8))
    return result


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
            
    if model_name in pretrainedmodels.model_names:
        backbone = pretrainedmodels.__dict__[model_name](pretrained=None)
        backbone = nn.Sequential(*list(backbone.children())[: -2])
    else:
        raise Exception('\nModel {} not exist'.format(model_name))
        
    solver = GAINSolver(backbone, num_classes, in_channels=in_channels)
    solver.load_model(checkpoint_path)
    
    black_box_model = pretrainedmodels.__dict__[black_box_model_name](pretrained=None)
    black_box_model = ClassifierNet(black_box_model, num_classes)
    black_box_model.load_state_dict(torch.load(black_box_model_path))
    black_box_model.eval()
    black_box_model = black_box_model.cuda()
    black_box_model = Classifier(black_box_model)
    
    black_box_model_test = pretrainedmodels.__dict__[black_box_model_name_test](pretrained=None)
    black_box_model_test = ClassifierNet(black_box_model_test, num_classes)
    black_box_model_test.load_state_dict(torch.load(black_box_model_path_test))
    black_box_model_test.eval()
    black_box_model_test = black_box_model_test.cuda()
    black_box_model_test = Classifier(black_box_model_test)
    
    images = np.array([np.asarray(Image.open(i).resize(image_size)) for i in data['image_path']])
    labels = np.array(data['label'])
    targets = np.array(data['target'])
    result = []
    for img, label, target in zip(images, labels, targets):
        original_image = img
        target_image = images[labels == target].copy()
        if targeted:
            tmp = Image.fromarray(original_image)
            best_norm = 255
            for t_m in target_image:
                adv = targeted_mask_attack(solver, 
                                           Image.fromarray(original_image), 
                                           Image.fromarray(t_m), label, target, black_box_model, max_iter)
                if adv is not None:
                    delta = (np.asarray(adv) - original_image).astype(np.float32)
                    norm = np.sqrt((delta ** 2).sum(-1)).mean()
                    if norm < best_norm:
                        tmp = adv
            result.append(tmp)
        else:
            result.append(non_targeted_mask_attack(solver, Image.fromarray(original_image), 
                                                   label, black_box_model, max_iter))
            
    if outputs_path is not None:
        if not os.path.exists(outputs_path):
            os.mkdir(outputs_path)
        for img, path in zip(result, data['image_path']):
            name = path.split('/')[-1]
            path = os.path.join(outputs_path, name)
            img.save(path)
    if with_test:
        cls = black_box_model_test.predict(result)
        cls = cls.argmax(axis=1)
        is_success = (targets == cls if targeted else labels != cls)
        result_np = np.array([np.asarray(i) for i in result])
        delta = (result_np - images).reshape(len(result_np), -1, 3).astype(np.float32)
        perturbation = np.sqrt((delta ** 2).sum(axis=-1)).mean(axis=-1)
        perturbation[~is_success] = 0
        score = (1 - is_success.mean()) * 64 + perturbation.mean()
        print('success: {} perturbation: {} score: {}'.format(is_success.mean(), perturbation.sum() / is_success.sum(), score))
        
    print(time.time() - start)