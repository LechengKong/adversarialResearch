import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import regularizers 
import util

model_name = "reg_cla_model.h5"
training_filenames = ["./pascal_train_and_half_val.record"]
val_filenames = ["./pascal_half_val.record"]
dir_head = tf.constant("../JPEGImages/")
learning_rate = 0.000001

train = tf.data.TFRecordDataset(training_filenames)

dataset_val = tf.data.TFRecordDataset(val_filenames)

train = train.repeat()  # Repeats dataset this # times
train = train.map(util.cla_reg_function_generator(dir_head))
train = train.batch(8)  # Batch size to use
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.map(util.cla_reg_function_generator(dir_head))
dataset_val = dataset_val.batch(8)

model = load_model(model_name, custom_objects={'iou_metric': util.iou_metric, 'iou_acc_metric' : util.iou_acc_metric})


model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
              loss={'classification_output': 'sparse_categorical_crossentropy',
                    'regression_output': 'mean_squared_error'},
              loss_weights={'classification_output': 0, 'regression_output': 1},
              metrics = {'classification_output' : 'accuracy', 'regression_output': [util.iou_metric, util.iou_acc_metric]}
              )

cp_callback = keras.callbacks.ModelCheckpoint(model_name, verbose=1)
early_stop_callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience=3)
model.fit(train.make_one_shot_iterator(),epochs=0,steps_per_epoch=1079,callbacks=[cp_callback, early_stop_callback])
ein = model.evaluate(train.make_one_shot_iterator(),steps=1079)
eout = model.evaluate(dataset_val.make_one_shot_iterator(),steps=364)
print(ein, eout)
