#coding=utf-8

with_transform = False
devices = [2, 3]
max_epoch = 300
batch_size = 32 * len(devices)
num_classes = 110
lr = 1e-2
interval = 10
epoch_size = 100
checkpoint_dir = 'saved_models'