#coding=utf-8

lr = 1e-1
devices = [3]
image_size = (224, 224)
num_classes = 110
epoch_size = 100
train_batch_size = 48
test_batch_size = 200
max_epoch = 50
checkpoint_path = None#'saved_models/best_model_Classifier.pt'
model_name = 'densenet121'
optimizer = 'sgd'
