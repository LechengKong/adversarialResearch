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

dir_head = tf.constant("../JPEGImagesBBin/")
filenames = tf.placeholder(tf.string, shape=[None])
resample_portion = 0.5
train_set_size = 5717
val_set_size = 5823
image_shape = (500,500,3)
learning_rate = 0.000005
layer_back = 3
layer_back_full = layer_back + 4
weight_decay = 0.0005
training_filenames = ["./pascal_train_and_half_val.record"]
val_filenames = ["./pascal_half_val.record"]
model_name = "vgg_5e-06_3_0.0005_train_val.h5"

reg_model = load_model(model_name)

reg_model.summary()

train = tf.data.TFRecordDataset(training_filenames)

dataset_val = tf.data.TFRecordDataset(val_filenames)

train = train.repeat()  # Repeats dataset this # times
train = train.shuffle(val_set_size-int(resample_portion*val_set_size)+train_set_size, seed = 87)
train = train.map(util.parse_function_generator(dir_head))
train = train.batch(8)  # Batch size to use
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.map(util.parse_function_generator(dir_head))
dataset_val = dataset_val.batch(8)

checkpoint_path = model_name
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)
early_stop_callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience=3)
reg_model.fit(train.make_one_shot_iterator(),epochs=0,steps_per_epoch=1079,callbacks=[cp_callback, early_stop_callback])
ein = reg_model.evaluate(train.make_one_shot_iterator(),steps=1079)
eout = reg_model.evaluate(dataset_val.make_one_shot_iterator(),steps=364)
print(ein)
print(eout)