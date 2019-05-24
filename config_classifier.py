#coding=utf-8

lr = 1e-2
devices = [1]
image_size = (224, 224)
crop_size = (224, 224)
num_classes = 110
epoch_size = 300
train_batch_size = 48
test_batch_size = 200
max_epoch = 50
checkpoint_path = None#'saved_models/best_model_Classifier.pt'
model_name = 'resnet50'
optimizer = 'sgd'
