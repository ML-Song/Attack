# coding:utf-8
import torchvision.transforms as T
from torchvision import utils, transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import os.path as osp
import sys
import torch
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class ImageDataSet(data.Dataset):
    def __init__(self, filepath, transform=None, return_path=False, with_size=False):
        assert osp.isfile(filepath), "{} no such file or directory".format(filepath)
        with open(filepath, 'r') as fp:
            info = json.load(fp)
        self.filepath = info['paths']
        self.return_path = return_path
#         with open(filepath, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 img = line.split()[0]
#                 assert is_image_file(img), "{} is not a image file".format(img)
#                 self.filepath.append(img)
#                 target.append(int(line.split()[1]))
        self.target = info['labels']
        if with_size:
            self.sizes = info['shapes']
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        image = pil_loader(self.filepath[idx])
        if self.transform:
            image = self.transform(image)
        if not self.return_path:
            return image, self.target[idx]
        else:
            return image, self.target[idx], self.filepath[idx]

# debug for dataset
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python {} [root_dir] [data.txt]".format(sys.argv[0]))
        sys.exit(1)
    dataset = ImageDataSet(sys.argv[1], sys.argv[2],
            T.Compose([
                T.Resize((256,128),interpolation=3),
                T.ToTensor()]))
    counter = 0
    # dataloader = data.DataLoader(dataset, batch_size=4)
    for img , label in dataset:
        print("img.size:", img.size())
        print("label:", label)
        counter += 1
        if counter == 4:
            break

