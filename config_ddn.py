#coding=utf-8

targeted = True
data_path = 'data/dev_data/'
devices = [2]
classifier_name = ['resnet50', 'vgg16', 'inceptionv4']
classifier_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in classifier_name]

batch_size = 16
steps = 100
use_post_process = True

num_classes = 110
output_dir = 'outputs/'
output_size = (299, 299)
