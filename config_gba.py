#coding=utf-8

devices = [0]
image_size = (224, 224)
num_classes = 110
classifier_name = ['resnet50', 'vgg16', 'inceptionv4']#, 'inceptionresnetv2']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]
batch_size = 16
