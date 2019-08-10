import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
import numpy as np
import util

tf.random.set_random_seed(87)

dir_head = tf.constant("../JPEGImages/")
filenames = tf.placeholder(tf.string, shape=[None])
resample_portion = 0.5
train_set_size = 5717
val_set_size = 5823
image_shape = (500,500,3)
learning_rate = 0.0001
layer_back = 3
layer_back_full = layer_back + 4
weight_decay = 0.0005
training_filenames = ["./pascal_train_and_half_val.record"]
val_filenames = ["./pascal_half_val.record"]
model_name = "vgg_5e-06_3_0.0005_train_val.h5"

train = tf.data.TFRecordDataset(training_filenames)
dataset_val = tf.data.TFRecordDataset(val_filenames)

train = train.repeat()  # Repeats dataset this # times
train = train.shuffle(val_set_size-int(resample_portion*val_set_size)+train_set_size, seed = 87)
train = train.map(util.reg_parse_function_generator(dir_head))
train = train.batch(8)  # Batch size to use
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.map(util.reg_parse_function_generator(dir_head))
dataset_val = dataset_val.batch(8)

reg_model = load_model(model_name)
box_model = Sequential()
for layer in reg_model.layers[:-5]:
    box_model.add(layer)
for layer in box_model.layers:
    layer.trainable = False
box_model.add(Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
box_model.add(Dropout(0.5))
box_model.add(Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
box_model.add(Dropout(0.5))
box_model.add(Dense(4))
box_model.summary()

box_model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics = [util.iou_metric])

checkpoint_path = "vgg_5e-reg.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)
early_stop_callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience=3)
box_model.fit(train.make_one_shot_iterator(),epochs=64,steps_per_epoch=1079,callbacks=[cp_callback, early_stop_callback])
# reg_model.fit_generator(train_iter,steps_per_epoch=715,epochs=64,callbacks=[cp_callback])
ein = box_model.evaluate(train.make_one_shot_iterator(),steps=1079)
eout = box_model.evaluate(dataset_val.make_one_shot_iterator(),steps=364)
print(ein)
print(eout)