import util
import os
import tensorflow as tf
input_dir = "../JPEGImages/"
output_dir = "../JPEGImagesBBin/"
filenames = ["./pascal_train_and_half_val.record", "./pascal_half_val.record"]

try:
    os.makedirs(output_dir)
except OSError:
    print ("directory exist")
else:
    print ("directory made")

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
    file = parsed_features["image/filename"]
    x = parsed_features['image/object/bbox/xmax'].values-parsed_features['image/object/bbox/xmin'].values
    y = parsed_features['image/object/bbox/ymax'].values-parsed_features['image/object/bbox/ymin'].values
    area = tf.math.multiply(x,y)
    index = tf.math.argmax(area)
    return file, [parsed_features["image/object/bbox/xmin"].values[index],
                    parsed_features["image/object/bbox/xmax"].values[index],
                    parsed_features["image/object/bbox/ymin"].values[index],
                    parsed_features["image/object/bbox/ymax"].values[index]]

train = tf.data.TFRecordDataset(filenames)

train = train.repeat(1)  # Repeats dataset this # times
train = train.map(parse_function)
train = train.batch(128)  # Batch size to use
iterator = train.make_one_shot_iterator()
val = iterator.get_next()
count = 0
total_area = 0
with tf.Session() as sess:
    try:
        while True:
            act_val = sess.run(val)
            for i, file in enumerate(act_val[0]):
                total_area += util.draw_black_box(file.decode("utf-8"), util.add_black_box_in_bound, act_val[1][i], input_dir, output_dir, 5000)
                count+=1
    except tf.errors.OutOfRangeError:
        pass

print(total_area/count)