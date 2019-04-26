#coding=utf-8

with_transform = False
devices = [2, 3]
max_epoch = 300
batch_size = 64 * len(devices)
num_classes = 110
lr = 1e-2
weight_decay = 5e-3
interval = 10
epoch_size = 100
checkpoint_dir = 'saved_models'