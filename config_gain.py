#coding=utf-8

lr = 1e-2
devices = [0, 1]
image_size = (224, 224)
num_classes = 110
epoch_size = 10
train_batch_size = 40
test_batch_size = 100
max_epoch = 50
checkpoint_path = 'saved_models/best_model_GAIN.pt'
loss_weights = (1, 1, 1)
area_threshold = 0.2