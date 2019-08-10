import os

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
#import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
import numpy as np

import util

dir_head = tf.constant("../JPEGImages/")
filenames = tf.placeholder(tf.string, shape=[None])
image_shape = (500,500,3)
learning_rate = 0.000005
layer_back = 3
weight_decay = 0.0005
training_filenames = ["./pascal_train.record"]
val_filenames = ["./pascal_val.record"]

tail_model = keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = image_shape, pooling = 'max')

for layer in tail_model.layers[:-layer_back]:
    layer.trainable = False

classification_model = Sequential()
classification_model.add(tail_model)
classification_model.add(Flatten())
classification_model.add(Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
classification_model.add(Dropout(0.5))
classification_model.add(Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
classification_model.add(Dropout(0.5))
classification_model.add(Dense(20, activation = 'softmax'))

classification_model.summary()

classification_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

train = tf.data.TFRecordDataset(training_filenames)
train = train.map(util.parse_function_generator(dir_head))
train = train.repeat()  # Repeats dataset this # times
train = train.shuffle(buffer_size=128)
train = train.batch(8)  # Batch size to use
dataset_val = tf.data.TFRecordDataset(val_filenames)
dataset_val = dataset_val.map(util.parse_function_generator(dir_head))
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.batch(8)

checkpoint_path = "vgg_"+str(learning_rate)+"_"+str(layer_back)+"_"+str(weight_decay)+"_"+str(time.time())+".h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)
early_stop_callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience=3)
classification_model.fit(train.make_one_shot_iterator(),epochs=64,steps_per_epoch=715,callbacks=[cp_callback, early_stop_callback])
# reg_model.fit_generator(train_iter,steps_per_epoch=715,epochs=64,callbacks=[cp_callback])
ein = classification_model.evaluate(train.make_one_shot_iterator(),steps=715)
eout = classification_model.evaluate(dataset_val.make_one_shot_iterator(),steps=728)
print(ein)
print(eout)