import sys

import numpy as np
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTableWidgetItem

from BHM_UI import Ui_MainWindow

RADIAN = np.pi / 180
# ------Gabor变换参数-----------
GABOR_SIZE = 5  # Gabor变换尺寸
sigma = 1.0
lambd = np.pi / 8
gamma = 0.5
psi = 0
# --------自适应阈值选取参数----------
THRESH_LIMIT = 80
THRESH_OFFSET = 25
# --------区域排查参数----------
MINMUM_AREA_WIDTH = 60
MAXIMUM_AREA_RATE = 0.8


class PyQtIpDemo(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.raw_pixmap=None
        self.labelImage.paintEvent = self.user_paint_event

    def btnOpenImage_Clicked(self):
        filename,_=QFileDialog.getOpenFileName(self,'打开图像',r'D:\教学\实习\2022\实习材料\jpg')
        if filename:
            self.captured = cv.imdecode(np.fromfile(filename, dtype=np.uint8), cv.IMREAD_UNCHANGED)
            self.raw_pixmap = QtGui.QPixmap(filename)

            self.raw_width = self.raw_pixmap.width()
            self.raw_height = self.raw_pixmap.height()
            window_width = self.labelImage.width()
            window_height = self.labelImage.height()
            self.scale_ratio = window_width / self.raw_width

            self.labelImage.update()

            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            self.original = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

    def btnGray_Clicked(self):
        self.src_f = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)

        rows, cols = self.src_f.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.src_f, cols, rows, bytesPerLine, QImage.Format_Indexed8))

        # print(self.src_f.shape)
        self.imagetoShow = self.src_f
        # self.imageShow()

        # self.btnGabor.setEnabled(True)
        # self.rect_exist_flag = False
        self.labelImage.update()

    def btnGabor_Clicked(self):
        if not hasattr(self, "src_f"):
            self.btnGray_Clicked()
        #  -----------------------------Gabor变换-----------------------
        # 首先生成Gabor核,方向为45度
        kernel = cv.getGaborKernel((GABOR_SIZE, GABOR_SIZE), sigma, 45 * RADIAN, lambd, gamma, psi, cv.CV_32F)
        gabor_img = cv.filter2D(self.src_f, cv.CV_32F, kernel)  # Gabor滤波
        self.gabor_img = cv.convertScaleAbs(gabor_img)  # 像素值转换为无符号整数UINT8

        rows, cols = self.gabor_img.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.gabor_img, cols, rows, bytesPerLine, QImage.Format_Indexed8))

        # self.imagetoShow = self.gabor_img
        # self.imageShow()

        # self.btnThreshold.setEnabled(True)
        self.labelImage.update()

    def btnThresh_Clicked(self):
        #   /*--------------------------自适应局部峰值-阈值分割----------------------------
        #   取图像直方图从0到最小像素值+THRESH_LIMIT之间最大值对应的灰度值，再加THRESH_OFFSET作为阈值
        #   这是本方法的一处特殊之处
        min1 = np.min(self.gabor_img)  # 得到像素最小值
        hist1 = cv.calcHist([self.gabor_img], [0], None, [256], [0.0, 255.0])  # 计算直方图
        th1 = np.argmax(hist1[0:min1 + THRESH_LIMIT], axis=0) + THRESH_OFFSET  #
        th1 = th1[0]
        _, self.thresh_img = cv.threshold(self.gabor_img, th1, 255, cv.THRESH_BINARY)  #

        self.imagetoShow = self.thresh_img
        # self.imageShow()
        rows, cols = self.thresh_img.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.thresh_img, cols, rows, bytesPerLine, QImage.Format_Indexed8))

        # self.imagetoShow = self.gabor_img
        # self.imageShow()

        # self.btnDeHole.setEnabled(True)
        self.labelImage.update()

    def btnDeHole_Clicked(self):
        strel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # 形态学操作的结构元素为直径为5的圆
        self.close_img = cv.morphologyEx(self.thresh_img, cv.MORPH_CLOSE, strel1)  # 目标为黑，闭运算能去除小的假目标

        rows, cols = self.close_img.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.close_img, cols, rows, bytesPerLine, QImage.Format_Indexed8))

        # self.imagetoShow = self.gabor_img
        # self.imageShow()

        # self.btnContour.setEnabled(True)
        self.labelImage.update()

    def user_paint_event(self, event):
        painter = QPainter(self.labelImage)
        if not (self.raw_pixmap is None):
            painter.drawPixmap(0, 0, int(self.raw_width * self.scale_ratio),
                               int(self.raw_height * self.scale_ratio), self.raw_pixmap)


if __name__=="__main__":
    '''app=QtWidgets.QApplication(sys.argv)
    window=PyQtIpDemo()
    window.show()
    sys.exit(app.exec_())'''
    smarties = cv.imread(r"jpg\2.jpg")
    gray_img = cv.cvtColor(smarties, cv.COLOR_BGR2GRAY)
    # 进行中值滤波
    img = cv.medianBlur(gray_img, 5)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 70, param1=100, param2=100, minRadius=100, maxRadius=0)
    # 对数据进行四舍五入变为整数
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画出来圆的边界
        cv.circle(smarties, (i[0], i[1]), i[2], (0, 0, 0), 2)
        # 画出来圆心
        cv.circle(smarties, (i[0], i[1]), 2, (0, 255, 255), 3)
    cv.imshow("Circle", smarties)
    cv.waitKey()
    cv.destroyAllWindows()

