import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import regularizers 
import util

model_name = "vgg_5e-06_3_0.0005_train_val.h5"
reg_model_name = "vgg_5e-reg.h5"
training_filenames = ["./pascal_train_and_half_val.record"]
dir_head = tf.constant("../JPEGImages/")
input_shape = (500, 500, 3)
weight_decay = 0.0005
learning_rate = 0.000005
base_model = load_model(model_name)
reg_model = load_model(reg_model_name, custom_objects={'iou_metric': util.iou_metric})
train = tf.data.TFRecordDataset(training_filenames)
train = train.repeat()  # Repeats dataset this # times
train = train.map(util.cla_reg_function_generator(dir_head))
train = train.batch(8)

vgg_input = Input(shape = input_shape, name = 'vgg16_input')

tail_model = base_model.layers[0]

base_layer = tail_model.layers[1](vgg_input)
for layer in tail_model.layers[2:]:
    layer.trainable = False
    base_layer = layer(base_layer)

base_layer = Flatten()(base_layer)

cla_layer = Dense(4096, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay))(base_layer)
cla_layer = Dropout(0.5)(cla_layer)
cla_layer = Dense(4096, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay))(cla_layer)
cla_layer = Dropout(0.5)(cla_layer)
cla_layer = Dense(20, activation = 'softmax', name = 'classification_output')(cla_layer)

reg_layer = Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay))(base_layer)
reg_layer = Dropout(0.5)(reg_layer)
reg_layer = Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(weight_decay))(reg_layer)
reg_layer = Dropout(0.5)(reg_layer)
reg_layer = Dense(4, name = 'regression_output')(reg_layer)

model = Model(inputs=vgg_input, outputs= [cla_layer, reg_layer])
for i, layer in enumerate(model.layers[21::2]):
    layer.set_weights(base_model.layers[2+i].get_weights())
    layer.trainable = False
for i, layer in enumerate(model.layers[22::2]):
    layer.set_weights(reg_model.layers[2+i].get_weights())
model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
              loss={'classification_output': 'sparse_categorical_crossentropy',
                    'regression_output': 'mean_squared_error'},
              loss_weights={'classification_output': 0, 'regression_output': 1},
              metrics = {'classification_output' : 'accuracy', 'regression_output': [util.iou_metric, util.iou_acc_metric]}
              )
model.summary()
checkpoint_path = 'reg_cla_model.h5'
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)
early_stop_callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience=3)
model.fit(train.make_one_shot_iterator(),epochs=32,steps_per_epoch=1079,callbacks=[cp_callback, early_stop_callback])
ein = model.evaluate(train.make_one_shot_iterator(),steps=1079)
