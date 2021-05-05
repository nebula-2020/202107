#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.save import *
import tensorflow as tf
from tensorflow.keras import *
import winsound
import time
from threading import *
MODEL_NAME = 'model'
sigmoid = tf.nn.sigmoid
relu = tf.nn.leaky_relu


def beeps(times: int = 3, frequency: int = 5000, duration: float = 1., delay: float = .6):
    for _ in range(abs(times)):
        winsound.Beep(max(37, min(32767, abs(frequency))),
                      int(abs(duration)*1000))
        time.sleep(abs(delay))


def beep(times: int = 3, frequency: int = 5000, duration: float = 1., delay: float = .6):
    t = Thread(target=beeps, args=(times, frequency, duration, delay))
    t.start()
    return t


def train():
    model = Sequential()
    model.add(layers.Dense(units=6, input_shape=[6], activation=relu))
    model.add(layers.Dense(units=18, input_shape=[6], activation=relu))
    model.add(layers.Dense(units=12, input_shape=[18], activation=relu))
    model.add(layers.Dense(units=6, input_shape=[12], activation=relu))
    model.add(layers.Dense(units=2, input_shape=[6], activation=sigmoid))
    opt = tf.optimizers.SGD(lr=0.1, decay=0, momentum=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy')

    class HaltCallback(callbacks.Callback):  # 自定义回调，给fit当作参数，没有回调也行
        last_avg = -1

        losts = []

        def on_epoch_end(self, epoch, logs={}):
            HaltCallback.losts.append(logs.get('loss'))
            avg = sum(HaltCallback.losts)/len(HaltCallback.losts)
            print(avg)
            if(logs.get('loss') <= 0.08 or (abs(avg - HaltCallback.last_avg) < 0.00001 and logs.get('loss') <= 0.15)):
                model.stop_training = True
            HaltCallback.last_avg = avg
            if(len(HaltCallback.losts) > 20):
                HaltCallback.losts.pop(0)

    cb = [HaltCallback()]
    x, y = [], []
    while x == [] or y == []:
        x, y = save.get_data()
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    history = model.fit(x, y, epochs=2000, verbose=1, callbacks=cb)
    # lost = history.history['loss']  # 获取损失
    model.save(MODEL_NAME)
    print('FINISH')
    beep()
    return model


def load():
    try:
        return tf.keras.models.load_model(MODEL_NAME)
    except:
        return None
