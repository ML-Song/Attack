#coding=utf-8

targeted = True
data_path = 'data/dev_data/'
devices = [1]
classifier_name = ['resnet50', 'vgg16', 'inceptionv4']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]

batch_size = 16
steps = 100

num_classes = 110
