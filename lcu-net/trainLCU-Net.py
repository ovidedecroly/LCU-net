import os

from keras.callbacks import ModelCheckpoint

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
from model import *
from pre_process import *

__label = '../Data/gt'
__source = '../Data/aug'

(trainX, trainY), (valX, valY), (testX, testY) = train_test_validation_split(__source, __label)
lcu_net = LCU_Net(debug=False)

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)


def train(lcu_net):
    h = lcu_net.fit(trainX, trainY,
                    shuffle=True,
                    batch_size=8,
                    validation_data=(valX, valY),
                    epochs=50,
                    callbacks=[model_checkpoint],
                    workers=-1)

    return h, lcu_net


