#coding=utf-8

targeted = False
data_path = 'data/dev_data/'
devices = [1]
classifier_name = ['resnet50', 'vgg16', 'inceptionv4']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]


image_size = (224, 224)
num_classes = 110
batch_size = 16
lr = 20 if targeted else 10
max_iteration = 10
patience = 3
max_perturbation = 10 if targeted else 5

