#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.bridge import Bridge
from PyQt5 import QtGui, QtWebChannel, QtWebEngineWidgets, QtWidgets
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
        self.setWindowTitle('摔机大王')
        self.resize(1280, 720)
        # pmp = QtGui.QPixmap('./assets/icons/icon.ico')
        icon = QtGui.QIcon()
        # icon.addPixmap(pmp, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
