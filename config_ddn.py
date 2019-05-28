#coding=utf-8

targeted = True
data_path = 'data/dev_data/'
devices = [2]
white_model_name = ['inceptionv4']
white_model_path = ['saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(i) for i in white_model_name]

black_model_name = 'resnet50'
black_model_path = 'saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(black_model_name)

batch_size = 110
steps = 100
use_post_process = True

num_classes = 110
output_dir = 'outputs/'
output_size = (299, 299)
image_mean = [0.5, 0.5, 0.5]
image_std = [0.5, 0.5, 0.5]

model_name = 'se_resnext50_32x4d'
in_channels = 2048
gain_model_path = 'saved_models/best_model_GAIN model: se_resnext50_32x4d optimizer: sgd loc: None temp: 0.1.pt'