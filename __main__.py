#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.form import Win
from core.bridge import Bridge
import sys
import os
import ctypes
from PyQt5 import QtCore, QtGui, QtWidgets


def init_splash():
    img = QtGui.QPixmap('./assets/images/welcome.png')
    img = img  .scaled(768, 432, QtCore.Qt.KeepAspectRatio)
    splash = QtWidgets.QSplashScreen(img)
    return splash


if __name__ == '__main__':
    # 隐藏控制台
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("itest")
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)
        ctypes.windll.kernel32.CloseHandle(whnd)
        pass
    #
    app = QtWidgets.QApplication(sys.argv)
    splash = init_splash()
    splash.show()
    # 可以显示启动信息
    # 关闭启动画面
    win = Win()
    # win.show()
    if splash is not None:
        splash.close()
    win.show()
    index = os.path.join(QtCore.QDir.currentPath(), 'assets/index.html')
    win.webView.load(QtCore.QUrl.fromLocalFile(index))
    sys.exit(app.exec_())
