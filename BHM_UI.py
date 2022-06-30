# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BHM_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnOpenImage = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenImage.setGeometry(QtCore.QRect(710, 10, 75, 23))
        self.btnOpenImage.setObjectName("btnOpenImage")
        self.labelImage = QtWidgets.QLabel(self.centralwidget)
        self.labelImage.setGeometry(QtCore.QRect(60, 30, 621, 491))
        self.labelImage.setObjectName("labelImage")
        self.btnGray = QtWidgets.QPushButton(self.centralwidget)
        self.btnGray.setGeometry(QtCore.QRect(710, 40, 75, 23))
        self.btnGray.setObjectName("btnGray")
        self.btnGabor = QtWidgets.QPushButton(self.centralwidget)
        self.btnGabor.setGeometry(QtCore.QRect(710, 70, 75, 23))
        self.btnGabor.setObjectName("btnGabor")
        self.btnThresh = QtWidgets.QPushButton(self.centralwidget)
        self.btnThresh.setGeometry(QtCore.QRect(710, 100, 75, 23))
        self.btnThresh.setObjectName("btnThresh")
        self.btnDeHole = QtWidgets.QPushButton(self.centralwidget)
        self.btnDeHole.setGeometry(QtCore.QRect(710, 130, 75, 23))
        self.btnDeHole.setObjectName("btnDeHole")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnOpenImage.clicked.connect(MainWindow.btnOpenImage_Clicked)
        self.btnGray.clicked.connect(MainWindow.btnGray_Clicked)
        self.btnGabor.clicked.connect(MainWindow.btnGabor_Clicked)
        self.btnThresh.clicked.connect(MainWindow.btnThresh_Clicked)
        self.btnDeHole.clicked.connect(MainWindow.btnDeHole_Clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnOpenImage.setText(_translate("MainWindow", "打开图像"))
        self.labelImage.setText(_translate("MainWindow", "显示图像"))
        self.btnGray.setText(_translate("MainWindow", "灰度化"))
        self.btnGabor.setText(_translate("MainWindow", "纹理滤波"))
        self.btnThresh.setText(_translate("MainWindow", "二值化"))
        self.btnDeHole.setText(_translate("MainWindow", "形态学滤波"))
