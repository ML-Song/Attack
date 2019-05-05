#coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import imp
import os
import csv
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import *
from scipy.misc import imresize
import numpy as np

def inception_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f).convert('RGB') as image:
            # image = np.asarray(image)
            # image = imresize(image, [224, 224]).astype(np.float32)
            image = image.resize((224, 224),Image.ANTIALIAS)
            # image = ( image / 255.0 ) * 2.0 - 1.0
            # image = Image.fromarray(np.uint8(image))
    return  image

def vgg_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f).convert('RGB') as image:
            # image = np.asarray(image)
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            # image = image.resize((224, 224),Image.ANTIALIAS)
            image = imresize(image, [224, 224]).astype(np.float32)
            image[:, :, 0] = image[:, :, 0] - _R_MEAN
            image[:, :, 1] = image[:, :, 1] - _G_MEAN
            image[:, :, 2] = image[:, :, 2] - _B_MEAN
            # image = Image.fromarray(np.uint8(image))
    return image

def test_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f).convert('RGB') as image:
            image = image.resize((299, 299),Image.ANTIALIAS)
            image = np.array(image).astype(np.float32)
    return image


if __name__ == '__main__':
    
    #parameters
    max_epoch = 30
    with_target = False
    beta = 8
    batch_size = 110
    num_classes = 110
    
    #transform
    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)

    model_dimension = 224
    center_crop = 224
    data_transform = transforms.Compose([
        # transforms.Resize(model_dimension),
        # transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])
    
    #model load
    MainModel = imp.load_source('MainModel', 'models/tf_to_pytorch_inception_v1.py')
    inception_model = torch.load('models/tf_to_pytorch_inception_v1.pth')
    inception_model = inception_model.cuda()
    inception_model.eval()
    inception_model.volatile = True
    print("inception_model is loaded!")
    
    MainModel = imp.load_source('MainModel', 'models/tf_to_pytorch_resnet_v1_50.py')
    resnet_model = torch.load('models/tf_to_pytorch_resnet_v1_50.pth')
    resnet_model = resnet_model.cuda()
    resnet_model.eval()
    resnet_model.volatile = True
    print("resnet_model is loaded!")
    
    MainModel = imp.load_source('MainModel', 'models/tf_to_pytorch_vgg16.py')
    vgg_model = torch.load('models/tf_to_pytorch_vgg16.pth')
    vgg_model = vgg_model.cuda()
    vgg_model.eval()
    vgg_model.volatile = True
    print("vgg_model is loaded!")

    #load origin image
    image_path = 'dev_data/'
    images = []
    with open(os.path.join(image_path, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(image_path, row['filename'])
            truelabel = int(row['trueLabel'])
            tragetedlabel = int(row['targetedLabel'])
            item = (filepath, truelabel, tragetedlabel)
            images.append(item)
    print("images is loaded!")

    #predict
    vgg_predictions = []
    resnet_predictions = []
    inception_predictions = []
    gt = []
    target_gt =[]
    vgg_score = 0.
    inception_score = 0.
    resnet_score = 0.
    for (each_image, each_label, each_target) in images:
        gt.append(each_label)
        target_gt.append(each_target)

        vgg_image = vgg_loader(each_image.replace("dev_data", "outputs"))
        vgg_input = data_transform(vgg_image)
        vgg_input = vgg_input.cuda()
        vgg_input = torch.unsqueeze(vgg_input, 0)

        inception_image = inception_loader(each_image.replace("dev_data", "outputs"))
        inception_input = data_transform(inception_image)
        inception_input = inception_input.cuda()
        inception_input = torch.unsqueeze(inception_input, 0)

        vgg_out = vgg_model(vgg_input).detach().cpu()
        vgg_out = vgg_out.argmax(dim=0).numpy().tolist()
        vgg_predictions.append(vgg_out)

        # resnet_out = resnet_model(vgg_input).detach().cpu()
        # resnet_out = resnet_out.argmax(dim=0).numpy().tolist()
        # resnet_predictions.append(resnet_out)

        inception_out = inception_model(inception_input).detach().cpu()
        inception_out = inception_out.argmax(dim=0).numpy().tolist()
        inception_predictions.append(inception_out)

        if each_label == vgg_out:
            vgg_score += 128
        else:
            test_input = test_loader(each_image)
            test_output = test_loader(each_image.replace("dev_data", "outputs"))
            vgg_score += np.sqrt(np.mean((test_input-test_output)**2))

        if each_label == inception_out:
            inception_score += 128
        else:
            test_input = test_loader(each_image)
            test_output = test_loader(each_image.replace("dev_data", "outputs"))
            inception_score += np.sqrt(np.mean((test_input-test_output)**2))

        # if each_label == resnet_out:
        #     resnet_score += 128
        # else:
        #     test_input = test_loader(each_image)
        #     test_output = test_loader(each_image.replace("dev_data", "output"))
        #     resnet_score += np.sqrt(np.mean((test_input-test_output)**2))

    print("vgg:{}, inception:{}, mean:{}".format(vgg_score/110, inception_score/110, (vgg_score+inception_score)/220))


