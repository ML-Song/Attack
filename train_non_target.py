#coding=utf-8
import imp
import csv
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from sklearn import metrics
from tensorboardX import SummaryWriter
from torchvision.datasets.folder import *
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import unet


class ImageListFolder(torch.utils.data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        images = []
        print(root)
        with open(os.path.join(root, 'dev.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = os.path.join(root, row['filename'])
                truelabel = int(row['trueLabel'])
                tragetedlabel = int(row['targetedLabel'])
                item = (filepath, truelabel, tragetedlabel)
                images.append(item)

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, attacked_target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, attacked_target

    def __len__(self):
        return len(self.imgs)
    
    
if __name__ == '__main__':
    max_epoch = 200
    with_target = False
    beta = 8
    batch_size = 32
    num_classes = 110
    interval = 10
    checkpoint_dir = 'saved_models'
    devices = [3]
    comment = 'model: {}, with_target: {}, beta: {}'.format('UNet', with_target, beta)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    MainModel = imp.load_source('MainModel', "models/tf_to_pytorch_inception_v1.py")
    pretrained_model = torch.load('models/tf_to_pytorch_inception_v1.pth')

    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)

    model_dimension = 224
    center_crop = 224
    data_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])

    # just use cpu for example
    pretrained_model = pretrained_model.cuda()
    pretrained_model.eval()
    pretrained_model.volatile = True
    
    attack_net = unet.UNet(3, 3, batch_norm=True).cuda()

    test_set = ImageListFolder(root='dev_data/', transform=data_transform)
    testing_data_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4, batch_size=batch_size,
                                         shuffle=True, drop_last=False)

    criterion_cls = nn.KLDivLoss(reduction='batchmean')
    optim = torch.optim.SGD(attack_net.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', verbose=True, patience=20, factor=0.2, threshold=5e-3)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with SummaryWriter(comment=comment) as writer:
        step = 0
        best_acc = 1
        for epoch in range(max_epoch):
            attack_net.train()
            for i, batch_data in tqdm.tqdm(enumerate(testing_data_loader)):
                batch_x = batch_data[0].cuda()
                batch_y = batch_data[1].cuda()
#                 batch_target = torch.ones((batch_y.size(0), num_classes)).cuda() / (num_classes)
                batch_target_one_hot = torch.FloatTensor(batch_y.size(0), num_classes).cuda()
                batch_target_one_hot.zero_()
                batch_target_one_hot.scatter_(1, batch_y.view(-1, 1), 1)
                batch_target = batch_target_one_hot
                optim.zero_grad()
                noise = attack_net(batch_x)
                batch_x_with_noise = batch_x + noise
                out = pretrained_model(batch_x_with_noise)
                out = 1 - torch.softmax(out, dim=1)
                out = torch.log(out)
                loss_cls = criterion_cls(out, batch_target)
                loss_min_noise = torch.sqrt((noise ** 2).mean())
#                 loss_min_noise = torch.abs(noise).mean()
                loss = (loss_cls + beta * loss_min_noise) / (1 + beta)
                loss.backward()
                optim.step()
            
                writer.add_scalar('loss_cls', loss_cls.data, global_step=step)
                writer.add_scalar('loss_min_noise', loss_min_noise.data, global_step=step)
                writer.add_scalar('loss', loss.data, global_step=step)
                writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=step)
                step += 1
            attack_net.eval()
            with torch.no_grad():
                right_num = 0
                score = 0
                gt = []
                predictions = []
                original_images = []
                noise_images = []
                for i, batch_data in tqdm.tqdm(enumerate(testing_data_loader)):
                    batch_x = batch_data[0].cuda()
                    batch_y = batch_data[1]
                    noise = attack_net(batch_x)
                    batch_x_with_noise = batch_x + noise
                    out = pretrained_model(batch_x_with_noise).detach().cpu()
                    gt.append(batch_y)
                    predictions.append(out.argmax(dim=1))
                    original_images.append(batch_x.detach().cpu())
                    noise_images.append(batch_x_with_noise.detach().cpu())
                    right_index = out.argmax(dim=1) == batch_y
                    wrong_index = out.argmax(dim=1) != batch_y
                    right_num += right_index.sum()
                    if wrong_index.sum() > 0:
                        score += torch.sqrt((((batch_x_with_noise - batch_x).detach().cpu()[wrong_index] * 128) ** 2).mean()) * wrong_index.sum()
                gt = torch.cat(gt).numpy()
                predictions = torch.cat(predictions).numpy()
                original_images = torch.cat(original_images)[: 200]
                noise_images = torch.cat(noise_images)[: 200]
                
                original_images = tv.utils.make_grid(
                    original_images, normalize=True, scale_each=True)
                noise_images = tv.utils.make_grid(
                    noise_images, normalize=True, scale_each=True)
                
                score += right_num * 128
                score /= len(test_set)
                scheduler.step(float(score))
                acc = metrics.accuracy_score(predictions, gt)
                writer.add_scalar('acc', acc, global_step=epoch)
                writer.add_scalar('score', score, global_step=epoch)
                if epoch % interval == 0:
                    writer.add_image('original_images', original_images, global_step=epoch)
                    writer.add_image('noise_images', noise_images, global_step=epoch)
            if best_acc > acc:
                best_acc = acc
                torch.save(attack_net.state_dict(), '{}/best_{}.pt'.format(checkpoint_dir, comment))