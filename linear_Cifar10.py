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


experiment_name = "CIFAR10_LINEAR_1_16_LR_0.5"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (-1, 32*32*3))/255.0

x_test = np.reshape(x_test, (-1, 32*32*3))/255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/'+experiment_name, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

linear_model = Sequential()
linear_model.add(Dense(10,
                       activation=sigmoid,
                       input_dim=(32*32*3)))



linear_model.compile(sgd(lr=0.5,),
                     mse, metrics=[categorical_accuracy])


linear_model.fit(x_train, y_train,
                 batch_size=8192,
                 epochs=10000,
                 verbose=1,
                 callbacks=[tb_callback],
                 validation_data=(x_test, y_test) )