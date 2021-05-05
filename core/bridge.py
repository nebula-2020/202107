from core.nn import load, train
from threading import Thread
from core.save import *
from PyQt5 import QtCore, QtGui, QtWebChannel, QtWebEngineWidgets, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as Qws


class Bridge(QtCore.QObject):
    def __init__(self, attr):
        super(Bridge, self).__init__(attr)
        self.__model = load()

    @QtCore.pyqtSlot(str)
    def save(self, j):
        json_array = json.loads(j)
        save.push(json_array)
        thread = Thread(target=save.save)
        thread.start()
        pass

    @QtCore.pyqtSlot(str, result=bool)
    def auto_control(self, j):
        json_array = json.loads(j)
        if self.__model:
            ret = self.__model.predict(Save.to_data(json_array))
            print('%s:: %s, %s:: %s' %
                  (j, ret[0][0], ret[0][1], ret[0][0] > ret[0][1]))
            ret = ret[0][0] > ret[0][1]
        else:
            return False
        return ret

    @QtCore.pyqtSlot()
    def train(self):
        self.__model = train()
