#coding=utf-8
with_target = True
with_transform = False
beta = 8
devices = [2, 3]
model_names = ['tf_to_pytorch_inception_v1', 'tf_to_pytorch_resnet_v1_50', 'tf_to_pytorch_vgg16']

max_epoch = 300
batch_size = 14 * len(devices)
num_classes = 110
lr = 1e-2
interval = 10
epoch_size = 100
checkpoint_dir = 'saved_models'
