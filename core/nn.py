#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.save import *
import tensorflow as tf
from tensorflow.keras import *
import matplotlib.pyplot as pyp

MODEL_NAME = 'model'


def train():
    model = Sequential()
    model.add(layers.Dense(units=6, input_shape=[6], activation='relu'))
    model.add(layers.Dense(units=12, input_shape=[6], activation='relu'))
    model.add(layers.Dense(units=12, input_shape=[12], activation='relu'))
    model.add(layers.Dense(units=2, input_shape=[12], activation='sigmoid'))
    opt = tf.optimizers.SGD(lr=0.1, decay=0, momentum=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy')

    class haltCallback(callbacks.Callback):  # 自定义回调，给fit当作参数，没有回调也行
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('loss') <= 0.1):  # 当损失充分小的时候停止学习
                self.model.stop_training = True

    cb = [haltCallback()]
    x, y = save.get_data(save.get_size()-1)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    history = model.fit(x, y, epochs=10000, verbose=1, callbacks=cb)
    lost = history.history['loss']  # 获取损失
    model.save(MODEL_NAME)
    pyp.plot(lost)
    pyp.show()


def load():
    try:
        return tf.keras.models.load_model(MODEL_NAME)
    except:
        return None
