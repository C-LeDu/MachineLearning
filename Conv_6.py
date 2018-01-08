from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.activations import *
from keras.losses import *
from keras.datasets import *
from keras.metrics import *
from keras.callbacks import *
import numpy as np
import keras


# /home/user/Documents/ESGI/Machine_Learning/Cifar-10/logs

experiment_name = "CIFAR10_CONVNET_6_DENSE_2_RELU_MAXPOOL_3_LR_0.5"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (-1, 32, 32, 3))/255.0

x_test = np.reshape(x_test, (-1, 32, 32, 3))/255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/'+experiment_name, histogram_freq=0,
                                          batch_size=32, write_graph=True, write_grads=False,
                                          write_images=False, embeddings_freq=0,
                                          embeddings_layer_names=None, embeddings_metadata=None)

conv_model = Sequential()

conv_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
conv_model.add(Activation('relu'))
conv_model.add(Conv2D(110, (3, 3)))
conv_model.add(Activation('relu'))
conv_model.add(MaxPool2D((3, 3)))
conv_model.add(Dropout(0.20))

conv_model.add(Conv2D(64, (3, 3)))
conv_model.add(Activation('relu'))
conv_model.add(Conv2D(64, (3, 3)))
conv_model.add(Activation('relu'))
conv_model.add(MaxPool2D((3, 3)))
conv_model.add(Dropout(0.20))


conv_model.add(Flatten())
conv_model.add(Dense(25, activation='relu'))
conv_model.add(Dense(10, activation='sigmoid'))

conv_model.compile(sgd(lr=0.5, ), mse, metrics=[categorical_accuracy])

conv_model.fit(x_train, y_train, batch_size=2000, epochs=2000, verbose=1, callbacks=[tb_callback],
               validation_data=(x_test, y_test), )

