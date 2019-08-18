import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import regularizers 
import util

model_name = "reg_cla_model.h5"
training_filenames = ["./pascal_train_and_half_val.record"]
val_filenames = ["./pascal_half_val.record"]
# dir_heads = [tf.constant("../JPEGImagesBB4574/"), tf.constant("../JPEGImagesBB5118/"), tf.constant("../JPEGImagesBB8937/"),
#                   tf.constant("../JPEGImagesBB12930/"), tf.constant("../JPEGImagesBB19476/"), tf.constant("../JPEGImagesBBin/")]
dir_heads = [tf.constant("../JPEGImages/")]
model = load_model(model_name, custom_objects={'iou_metric': util.iou_metric, 'iou_acc_metric' : util.iou_acc_metric})

train = tf.data.TFRecordDataset(training_filenames)

test = tf.data.TFRecordDataset(val_filenames)
eins = []
eouts = []
for dir_head in dir_heads:
      train_temp = train.repeat()  # Repeats dataset this # times
      train_temp = train_temp.map(util.cla_reg_function_generator(dir_head))
      train_temp = train_temp.batch(8)  # Batch size to use
      test_temp = test.repeat()
      test_temp = test_temp.map(util.cla_reg_function_generator(dir_head))
      test_temp = test_temp.batch(8)
      ein = model.evaluate(train_temp.make_one_shot_iterator(),steps=1079)
      eout = model.evaluate(test_temp.make_one_shot_iterator(),steps=364)
      eins.append(ein)
      eouts.append(eout)


print(eins)

print(eouts)
