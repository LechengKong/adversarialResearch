import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
tf.random.set_random_seed(87)

dir_head = tf.constant("../JPEGImages/")

training_filenames = ["./pascal_train.record"]
val_filenames = ["./pascal_val.record"]
new_trainset_filename = "pascal_train_and_half_val.record"
new_valset_filename = "pascal_half_val.record"
new_valset_used = ["./pascal_half_val.record"]
new_trainset_used = ["./pascal_train_and_half_val.record"]
resample_portion = 0.5
train_set_size = 5717
val_set_size = 5823
def parse_function(example_proto):
    feature={
    'image/height': tf.FixedLenFeature((), tf.int64, -1),
    'image/width': tf.FixedLenFeature((), tf.int64, -1),
    'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/class/text': tf.VarLenFeature(tf.string),
    'image/object/class/label': tf.VarLenFeature(tf.int64),
    'image/object/difficult': tf.VarLenFeature(tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, feature)
    file = tf.string_join([dir_head, parsed_features["image/filename"]])
    x = parsed_features['image/object/bbox/xmax'].values-parsed_features['image/object/bbox/xmin'].values
    y = parsed_features['image/object/bbox/ymax'].values-parsed_features['image/object/bbox/ymin'].values
    area = tf.math.multiply(x,y)
    index = tf.math.argmax(area)
    image_string = tf.read_file(file)
    image = tf.image.decode_image(image_string, channels=3,dtype=tf.uint8)
    image.set_shape([None, None, None])
    fwidth = tf.to_float(parsed_features["image/width"])
    fheight = tf.to_float(parsed_features["image/height"])
    hgap = (500-fwidth)/2
    vgap = (500-fheight)/2
    nxmin = (parsed_features["image/object/bbox/xmin"].values[index]*fwidth + hgap)/500.0
    nxmax = (parsed_features["image/object/bbox/xmax"].values[index]*fwidth + hgap)/500.0
    nymin = (parsed_features["image/object/bbox/ymin"].values[index]*fheight + vgap)/500.0
    nymax = (parsed_features["image/object/bbox/ymax"].values[index]*fheight + vgap)/500.0
    image = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
    image = preprocess_input(image)
    #label = tf.cast(parsed_features["image/object/class/label"], tf.float32)
    #label = tf.cast(tf.reshape(parsed_features["image/object/class/label"], shape=[]), dtype=tf.int32)
    return {'vgg16_input':image}, [nxmin, nxmax, nymin, nymax]

train_writer = tf.python_io.TFRecordWriter(new_trainset_filename)
test_writer = tf.python_io.TFRecordWriter(new_valset_filename)
train = tf.data.TFRecordDataset(training_filenames)
val = tf.data.TFRecordDataset(val_filenames)

val = val.shuffle(val_set_size, seed = 87)
dataset_val = val.take(int(resample_portion*val_set_size))
train = train.concatenate(val.skip(int(resample_portion*val_set_size)))

train = train.repeat(1)  # Repeats dataset this # times
train = train.batch(128)  # Batch size to use

dataset_val = dataset_val.repeat(1)
dataset_val = dataset_val.batch(128)

iterator_val = dataset_val.make_one_shot_iterator()
next_val = iterator_val.get_next()
iterator_train = train.make_one_shot_iterator()
next_train = iterator_train.get_next()
with tf.Session() as sess:
    try:
        while True:
            act_val = sess.run(next_val)
            for pic in act_val:
                test_writer.write(pic)
    except tf.errors.OutOfRangeError:
        pass
    try:
        while True:
            act_train = sess.run(next_train)
            for pic in act_train:
                train_writer.write(pic)
    except tf.errors.OutOfRangeError:
        pass

test_writer.close()
train_writer.close()