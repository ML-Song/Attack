#coding=utf-8

# inputs_path = 'data/dev_data/'
# outputs_path = 'outputs/'

targeted = True
model_name = 'se_resnext50_32x4d'
checkpoint_path = 'saved_models/best_model_GAIN model: se_resnext50_32x4d optimizer: sgd loc: None temp: 0.1.pt'
image_size = (299, 299)
num_classes = 110
in_channels = 2048
max_iter = 5

black_box_model_name = 'inceptionv4'
black_box_model_path = 'saved_models/best_model_Classifier model: {} optimizer: sgd.pt'.format(black_box_model_name)

with_test = True
black_box_model_name_test = 'inceptionv4'
black_box_model_path_test = 'saved_models/best_model_Classifier model: {} with_augmentation: True.pt'.format(black_box_model_name_test)
