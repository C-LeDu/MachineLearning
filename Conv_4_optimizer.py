from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.activations import *
from keras.losses import *
from keras.datasets import *
from keras.metrics import *
#from keras.callbacks import *
import numpy as np
import keras


experiment_name="_CONVNET_3_32_64_32_RELU_3_MAXPOOL_3_2_DROPOUT_3_RELU_64_LR_1_OPTI_SGDM"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (-1, 32, 32, 3))/255.0
x_test = np.reshape(x_test, (-1, 32, 32, 3))/255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

tb_callback = keras.callbacks.TensorBoard(".\logs" + experiment_name)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.10))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


model.compile(keras.optimizers.rmsprop(lr=1, ), mse,
              callbacks=[tb_callback],
              metrics=[categorical_accuracy])


model.fit(x_train, y_train,
          batch_size=4096,
          epochs=500,
          verbose=1, callbacks=[tb_callback],
          validation_data=(x_test, y_test), )
