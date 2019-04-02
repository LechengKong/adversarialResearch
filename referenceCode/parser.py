import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
#import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
import numpy as np
weight_decay = 0.0005
dir_head = tf.constant("../JPEGImages/")
one = tf.constant(1, dtype=tf.int64)
filenames = tf.placeholder(tf.string, shape=[None])
image_shape = (500,500,3)
training_filenames = ["./pascal_train.record"]

def imgs_input_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(example_proto):
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
        image_string = tf.read_file(file)
        image = tf.image.decode_image(image_string, channels=3,dtype=tf.uint8)
        image.set_shape([None, None, None])
        image = preprocess_input(image)
        image = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
        label = tf.cast(parsed_features["image/object/class/label"], tf.float32)
        #label = tf.cast(tf.reshape(parsed_features["image/object/class/label"], shape=[]), dtype=tf.int32)
        return {'vgg16_input':image}, [label.values[0]-1]
    
    dataset = tf.data.TFRecordDataset(filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels



#sanity check
# training_filenames = ["./pascal_train.record"]
# next_batch = imgs_input_fn(training_filenames,batch_size=128)
# with tf.Session() as sess:
#     first_batch = sess.run(next_batch)
# print(first_batch)


def build_tail(tail, head, tail_trainable = 0, head_trainable = 1):
    if head_trainable:
        head.trainable = True
    else:
        head.trainable = False
    if tail_trainable:
        tail.trainable = True
    else:
        tail.trainable = False
    model = keras.Sequential()
    model.add(tail)
    model.add(head)
    return model


tail_model = keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = image_shape, pooling = 'max', classes = 20)

reg_head = keras.Sequential()
reg_head.add(Flatten())
reg_head.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay), activation = 'relu'))
reg_head.add(Dropout(0.5))
reg_head.add(Dense(20, activation = 'softmax'))

reg_model = build_tail(tail_model,reg_head, tail_trainable=1)
reg_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# #keras sanity check
# toy = np.random.random((40,500,500,3))
# toy_label = np.random.randint(20,size = 40)
# reg_model.fit(toy,toy_label,batch_size=8)
model_dir = os.path.join(os.getcwd(), "models\\classhead")
os.makedirs(model_dir, exist_ok=True)
print("model_dir: ",model_dir)
est_classreg = tf.keras.estimator.model_to_estimator(keras_model=reg_model,
                                                    model_dir=model_dir)
# predict_results = est_classreg.predict(
#     input_fn=lambda: imgs_input_fn(training_filenames, 
#                                    perform_shuffle=False,
#                                    batch_size=10))
est_classreg.train(input_fn=lambda: imgs_input_fn(training_filenames, batch_size=8))