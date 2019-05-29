#coding=utf-8

lr = 1e-2
devices = [1]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 26
test_batch_size = 50
max_epoch = 50
classifier_name = ['inceptionv4']#, 'inceptionresnetv2']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]

test_classifier_name = ['resnet50']#, 'inceptionresnetv2']
test_classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) 
                        for i in test_classifier_name]

optimizer = 'sgd'
targeted = False
checkpoint_path = 'saved_models/best_model_Attack targeted: False beta: 8 with_gain: False as_noise: True.pt'
weight = 64
loss_mode = ('margin', 'cross_entropy')[1]
beta = 8

gain_model_name = None#'se_resnext50_32x4d'
in_channels = 2048
gain_checkpoint_path = 'saved_models/best_model_GAIN model: se_resnext50_32x4d optimizer: sgd loc: None temp: 0.1.pt'

as_noise = True
inference_batch_size = 16
