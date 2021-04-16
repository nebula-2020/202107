#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.bridge import Bridge
import sys
import os
import ctypes
from PyQt5 import QtCore, QtGui, QtWebChannel, QtWebEngineWidgets, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as Qws


class Win(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.python_bridge = Bridge(None)
        self.setContentsMargins(0, 0, 0, 0)
##====##====##====##====##====##====##====##====##====##====##====##====##
        self.webView = QtWebEngineWidgets.QWebEngineView()
        self.webView.setContentsMargins(0, 0, 0, 0)
        self.webView.settings().setAttribute(Qws.JavascriptEnabled, True)
        channel = QtWebChannel.QWebChannel(self.webView.page())
        self.webView.page().setWebChannel(channel)
        channel.registerObject("py", self.python_bridge)
##====##====##====##====##====##====##====##====##====##====##====##====##
        layout = QtWidgets. QVBoxLayout()
        layout.addWidget(self.webView)
        layout.setContentsMargins(0, 0, 0, 0)
##====##====##====##====##====##====##====##====##====##====##====##====##
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
##====##====##====##====##====##====##====##====##====##====##====##====##
        self.setWindowTitle('Helicopter')
        # pmp = QtGui.QPixmap('./assets/icons/icon.ico')
        icon = QtGui.QIcon()
        # icon.addPixmap(pmp, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)


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
    splash = QtWidgets.QSplashScreen(
        QtGui.QPixmap('./assets/images/welcome.png'))
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
