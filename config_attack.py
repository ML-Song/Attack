#coding=utf-8

lr = 1e-2
devices = [1]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 12
test_batch_size = 24
max_epoch = 50
classifier_name = ['vgg16']#, 'inceptionresnetv2']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]
optimizer = 'sgd'
targeted = False
checkpoint_path = 'saved_models/best_model_Attack targeted: True weight: 64 loss_mode: margin max_l2: 64.pt'
weight = 64
loss_mode = ('margin', 'cross_entropy')[1]
