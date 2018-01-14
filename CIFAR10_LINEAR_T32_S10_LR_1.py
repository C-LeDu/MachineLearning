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


experiment_name="_LINEAR_T32_S10_LR_1"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



x_train = np.reshape(x_train,(-1,32*32*3))/255.0
x_test = np.reshape(x_test,(-1,32*32*3))/255.0


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

tb_callback = TensorBoard("E:\\Documents\logs_deep_learning\CIFAR10" + experiment_name)

linear_model = Sequential()
linear_model.add(Dense(32,
                       activation='tanh',
                       input_dim=(32*32*3)))
linear_model.add(Dense(10, activation='sigmoid'))



linear_model.compile(sgd(lr=1),
                     mse, metrics=[categorical_accuracy])


linear_model.fit(x_train, y_train,
                 batch_size=8192,
                 epochs=2000,
                 verbose=1,
                 callbacks=[tb_callback],
                 validation_data=(x_test,y_test), )
