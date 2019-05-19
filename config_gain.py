#coding=utf-8

lr = 1e-3
devices = [0]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 16#48
test_batch_size = 50
max_epoch = 50
checkpoint_path = None#'saved_models/best_model_GAIN model: resnet50 optimizer: sgd.pt'
loss_weights = (1, 1)
area_threshold = 0.25
in_channels = 512#2048
model_name = 'drn_d_54'#'resnet50'
optimizer = 'sgd'
loc = None#-0.2
