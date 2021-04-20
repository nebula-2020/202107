#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from threading import *
import tensorflow as tf
import traceback


class Save:
    __PLAYER_Y = 'y'
    __VEC = 'v'
    __DOOR_TOP = 't'
    __DISTANCE = 'd'
    __DOOR_BOTTOM = 'b'
    __LABEL = 'l'
    __TAGS = [__PLAYER_Y, __VEC, __DOOR_TOP,
              __DISTANCE, __DOOR_BOTTOM, __DISTANCE, __LABEL]

    def __init__(self):
        self.__folder = '.'
        self.__fileName = 'save.json'
        self.__data = []
        self.__index = 0
        path = os.path.join(self.__folder, self.__fileName)
        if os.path.exists(path):
            try:
                with tf.io.gfile.GFile(path, 'r') as file:
                    data = file.read()
                    self.__data = json.loads(data)
            except:
                pass
            pass
        pass
    pass

    def save(self):
        if not os.path.exists(self.__folder):
            os.makedirs(self.__folder)
            pass
        path = os.path.join(self.__folder, self.__fileName)
        with tf.io.gfile.GFile(path, 'w') as file:
            data = json.dumps(self.__data)
            print('SAVE: %d' % len(self.__data))
            file.write(data)
            pass
        pass

    def push(self, data: list):
        length = len(Save.__TAGS)
        for ele in data:
            count = 0
            new = {}
            for tag in Save.__TAGS:
                t = type(ele[tag])
                if t is float or t is int:
                    count += 1
                    new[tag] = ele[tag]
                else:
                    break
            if count == length:
                self.__data.append(ele)
        pass

    def get_size(self) -> int:
        return len(self.__data)

    def get_data(self, size: int) -> tuple:
        x = []
        y = []
        if len(self.__data) > 2:
            end = self.__index+abs(size)
            end = end % len(self.__data)
            xtags = [e for e in Save.__TAGS]
            xtags.remove(Save.__LABEL)
            x_tag_len = len(xtags)
            for ele in self.__data[self.__index: end]:
                res = []
                for tag in xtags:
                    res.append(ele[tag])
                    pass
                if len(res) != x_tag_len:
                    print(y)
                x.append(res)
                y.append([1, 0]if ele[Save.__LABEL] > .5 else[0, 1])
                pass
            self.__index = end
        return x, y

    @staticmethod
    def to_data(j: dict) -> list:
        x = []
        xtags = [e for e in Save.__TAGS]
        xtags.remove(Save.__LABEL)
        length = len(Save.__TAGS)
        count = 0
        for ele in j:
            new = {}
            for tag in Save.__TAGS:
                t = type(ele[tag])
                if t is float or t is int:
                    count += 1
                    new[tag] = ele[tag]
                else:
                    break

        res = []
        if count == length:
            for tag in xtags:
                res.append(ele[tag])
                pass
            x.append(res)
            pass
        x.append(res)
        return x


save = Save()
