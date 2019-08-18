import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras
import imageio
import random
import math
import numpy as np

def make_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with tf.Session() as sess:
        while True:
            yield sess.run(next_val)

def iou_metric(y_true, y_pred):
    y_true_t = tf.transpose(y_true)
    y_pred_t = tf.transpose(y_pred)
    intersect_xmin = tf.maximum(y_true_t[0], y_pred_t[0])
    intersect_xmax = tf.minimum(y_true_t[1], y_pred_t[1])
    intersect_ymin = tf.maximum(y_true_t[2], y_pred_t[2])
    intersect_ymax = tf.minimum(y_true_t[3], y_pred_t[3])
    intersect_x = tf.maximum(intersect_xmax - intersect_xmin, 0)
    intersect_y = tf.maximum(intersect_ymax - intersect_ymin, 0)
    intersect = intersect_x * intersect_y
    union = (y_true_t[1] - y_true_t[0]) * (y_true_t[3] - y_true_t[2]) + (tf.abs(y_pred_t[1] - y_pred_t[0])) * (tf.abs(y_pred_t[3] - y_pred_t[2])) - intersect
    iou = intersect/union
    return iou

def iou_acc_metric(y_true, y_pred):
    iou = iou_metric(y_true, y_pred)
    acc = tf.clip_by_value(tf.sign(iou - 0.5), 0, 1)
    return acc

def joint_acc_metric(y_true, y_pred):
    y_cla_pred, y_reg_pred = tf.split(y_pred, [20, 4], 1)
    y_cla_true, y_reg_true = tf.split(y_true, [1, 4], 1)
    y_cla_true = tf.squeeze(y_cla_true, -1)
    y_label_pred = tf.argmax(y_cla_pred, axis = -1)
    cla_acc = tf.equal(y_cla_true, tf.to_float(y_label_pred))
    iou = iou_metric(y_reg_true, y_reg_pred)
    reg_acc = tf.greater(iou, 0.5)
    return tf.cast(tf.logical_and(cla_acc, reg_acc), tf.float64)

def naive_loss(y_true, y_pred):
    return tf.constant(1.0)

def parse_function_generator(dir_head):
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
        image = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
        image = preprocess_input(image)
        #label = tf.cast(parsed_features["image/object/class/label"], tf.float32)
        #label = tf.cast(tf.reshape(parsed_features["image/object/class/label"], shape=[]), dtype=tf.int32)
        return {'vgg16_input':image}, [parsed_features["image/object/class/label"].values[index]-1]
    return parse_function

def reg_parse_function_generator(dir_head):
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
    return parse_function

def cla_reg_function_generator(dir_head):
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
        return {'vgg16_input':image}, {'classification_output' : [parsed_features["image/object/class/label"].values[index]-1],
                                       'regression_output' : [nxmin, nxmax, nymin, nymax]}
    return parse_function

def joint_parse_function_generator(dir_head):
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
        return {'vgg16_input':image}, {'joint_output' : [tf.to_float(parsed_features["image/object/class/label"].values[index]-1),
                                                            nxmin, nxmax, nymin, nymax]}
    return parse_function

def fileset_repartition(dataset_dirs, size, new_trainset_filename, new_testset_filename,train_portion = 0.75, seed = None):
    train_writer = tf.python_io.TFRecordWriter(new_trainset_filename)
    test_writer = tf.python_io.TFRecordWriter(new_testset_filename)
    all_data = tf.data.TFRecordDataset(dataset_dirs)
    all_data.shuffle(size, seed = 87)
    train_set = all_data.take(int(size * train_portion))
    test_set = all_data.skip(int(size * train_portion))
    train_set.repeat(1)
    test_set.repeat(1)
    train_set.batch(128)
    test_set.batch(128)
    iterator_test = test_set.make_one_shot_iterator()
    next_test = iterator_test.get_next()
    iterator_train = train_set.make_one_shot_iterator()
    next_train = iterator_train.get_next()
    with tf.Session() as sess:
        try:
            while True:
                act_val = sess.run(next_test)
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

def dataset_distribution_check(dataset):
    iterator = dataset.make_one_shot_iterator()
    dmap = [0] * 20
    next_val = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                act_val = sess.run(next_val)
                for i in act_val[1]:
                    dmap[i[0]]+=1
        except tf.errors.OutOfRangeError:
            pass
    print(dmap)

def blackb_parse_function_generator(dir_head):
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
    return parse_function

def add_black_box(original_box, image_size, box_size, random_seed = None):
    box_w = int(math.ceil(max(random.random()*image_size[1], box_size / image_size[0])))
    box_h = int(box_size/box_w)
    if box_h > image_size[0]:
        return None
    inbound_xmin = int(max(original_box[0] * image_size[1] - box_w, 0))
    inbound_ymin = int(max(original_box[2] * image_size[0] - box_h, 0))
    inbound_xmax = int(original_box[1] * image_size[1])
    inbound_ymax = int(original_box[3] * image_size[0])
    random_range_y = image_size[0] - box_h - (inbound_ymax - inbound_ymin)
    if random_range_y < 0:
        random_range_x = image_size[1] - box_w - (inbound_xmax - inbound_xmin)
        if random_range_x < 0:
            return None
        x = random.random() * random_range_x
        if x > inbound_xmin:
            x += inbound_xmax - inbound_xmin
        y = random.random() * (image_size[0]-box_h)
    else:
        x = random.random() * (image_size[1]-box_w)
        if x < inbound_xmax and x > inbound_xmin:
            y = random.random() * random_range_y
            if y > inbound_ymin:
                y += inbound_ymax - inbound_ymin
        else:
            y = random.random() * (image_size[0] - box_h)
    return int(x), int(y), box_w, box_h

def add_black_box_in_bound(original_box, image_size, box_size, random_seed = None):
    inbound_xmin = int(original_box[0] * image_size[1])
    inbound_ymin = int(original_box[2] * image_size[0])
    inbound_xmax = int(original_box[1] * image_size[1])
    inbound_ymax = int(original_box[3] * image_size[0])
    inbound_x = inbound_xmax - inbound_xmin
    inbound_y = inbound_ymax - inbound_ymin
    box_w = int(math.ceil(max(random.random()*inbound_x, box_size / inbound_y)))
    box_h = int(box_size/box_w)
    if box_h > inbound_y:
        return None
    x = random.random() * (inbound_xmax - inbound_xmin) + inbound_xmin
    y = random.random() * (inbound_xmax - inbound_xmin) + inbound_ymin
    return int(x), int(y), box_w, box_h

def draw_black_box(filename, generate_box, original_box, input_dir, output_dir, box_area):
    img = imageio.imread(input_dir + filename)
    img_shape = np.shape(img)
    black_box = generate_box(original_box, img_shape, box_area)
    retrial = 10
    while black_box == None and retrial > 0:
        black_box = generate_box(original_box, img_shape, box_area)
        retrial -= 1
    if retrial == 0:
        imageio.imsave(output_dir + filename, img)
        return 0
    y_upper = min(black_box[1] + black_box[3], img_shape[0])
    x_upper = min(black_box[0] + black_box[2], img_shape[1])
    for j in range(black_box[1], y_upper):
        for i in range(black_box[0], x_upper):
            img[j][i] = 0
    imageio.imsave(output_dir + filename, img)
    return (x_upper - black_box[0]) * (y_upper - black_box[1])