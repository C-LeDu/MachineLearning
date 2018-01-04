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


experiment_name = "CIFAR10_CONVNET_1_16_MAXPOOL_1_2_LR_0.5"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (-1, 32, 32, 3))/255.0

x_test = np.reshape(x_test, (-1, 32, 32, 3))/255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/'+experiment_name, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

linear_model = Sequential()
linear_model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)))
linear_model.add(MaxPool2D((3, 3)))
linear_model.add(Flatten())
linear_model.add(Dense(10, activation='sigmoid'))
# 10 tanh 28*28
# meilleur resultat sur la fonction sigmoid puis sur la hard sigmoid et enfin la tanh
linear_model.compile(sgd(lr=1.5,), mse, metrics=[categorical_accuracy]) # LR=0.01

# linear_model.fit(x_train, y_train, batch_size=1000, epochs=10000,callbacks=[tb_callback], verbose=1,
# validation_data=(x_test,y_test),)#batch_size=4096 (la ram utilisé)
# --- epochs=10000 (le nombre de fois que l'on recommence les tests sur les elements d'entrainements)
linear_model.fit(x_train, y_train, batch_size=8000, epochs=2000, verbose=1, callbacks=[tb_callback], validation_data=(x_test, y_test),)

