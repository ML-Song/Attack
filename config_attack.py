#coding=utf-8

lr = 1e-3
devices = [2, 3]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 12
test_batch_size = 24
max_epoch = 50
classifier_name = ['inceptionresnetv2']#, 'inceptionresnetv2']
classifier_path = ['saved_models/best_model_Classifier model: {}.pt'.format(i) for i in classifier_name]
optimizer = 'sgd'
targeted = True
checkpoint_path = None
weight = 1
