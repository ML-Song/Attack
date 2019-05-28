#coding=utf-8

lr = 1e-3
devices = [3]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 24#48
test_batch_size = 200
max_epoch = 50
checkpoint_path = None#'saved_models/best_model_GAIN model: se_resnet50 optimizer: sgd loc: -0.3 temp: 0.1.pt'
loss_weights = (1, 1)
area_threshold = 0.25
in_channels = 2048#512#
model_name = 'se_resnext50_32x4d'#'drn_d_54'#
optimizer = 'sgd'
loc = None#-0.3#None#
temp = 0.1
