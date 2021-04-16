from core.nn import load, train
from threading import Thread
from core.save import *
import sys
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
        ret = False
        if self.__model:
            ret = self.__model.predict(Save.to_data(json_array))
        print(ret)
        return True if ret[0][0] > ret[0][1] else False

    @QtCore.pyqtSlot()
    def train(self):
        train()

    # @QtCore.pyqtSlot(result=str)
    # def next(self):
    #     hi = None
    #     try:
    #         if len(self.__history) >= 30:
    #             hi = self.__history.pop(
    #                 random.randint(0, len(self.__history)-1))
    #     except:
    #         pass
    #     if random.random() > .75:
    #         data = self.__save.get_data()
    #         data_keys = list(data.keys())
    #         ran = random.randint(0, len(data_keys)-1)
    #         select = data[data_keys[ran]][Save.PASS]
    #         select_keys = select.keys()
    #         ran = random.randint(0, len(select_keys)-1)
    #         word = list(select_keys)[ran]
    #         w = self.__save.get_weight(word)
    #         if w > .75:
    #             if hi is not None:
    #                 word = hi
    #         ret = self.__paper.quest_one(word)
    #     else:
    #         ret = self.__paper.next()
    #     try:
    #         self.__history.append(ret[TEXT])
    #     except:
    #         pass
    #     ret = json.dumps(ret, ensure_ascii=False)
    #     return ret

    # @QtCore.pyqtSlot(str, result=str)
    # def answer(self, ans):
    #     try:
    #         j = json.loads(ans)
    #         t = j[Py.__TEXT]
    #         a = j[Py.__ANSWER]
    #         return self.__paper.answer(t, a)
    #     except:
    #         traceback.print_exc()
    #         return '{}'
