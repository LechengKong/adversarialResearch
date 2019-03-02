from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np

imgshape = (224, 224, 3)

conv_out_shape = (7, 7, 512)

m = np.zeros(imgshape)
model_vgg16_conv = VGG16(weights = None, input_shape=imgshape, include_top = False)
model_vgg16_conv.summary()

fc_reg = keras.Sequential()

fc_reg.add(input_l)


input_l = Input(shape=conv_out_shape,name = 'image_input')
train_set = model_vgg16_conv.predict(m.reshape((1,224,224,3)))

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

print(train_set.shape)
