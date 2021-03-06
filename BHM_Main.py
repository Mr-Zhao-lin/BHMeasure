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


class PyQtIpDemo(QMainWindow,Ui_MainWindow):  #多继承
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.raw_pixmap=None
        self.labelImage.paintEvent = self.user_paint_event

    def circle_judge(self,circle, img):
        circle_r=circle[2]
        area_circle=np.pi*np.power(circle_r,2)
        num_points=round(area_circle/10)  # 使用散点抽样来估计面积
        Rho=0  # 圆得极坐标参数1
        theta=0  # 极坐标参数之角度
        Rho=np.sqrt(np.random.uniform(0, circle_r, num_points))*np.sqrt(circle_r)
        theta=np.random.uniform(0, 2*np.pi, num_points)
        points_x=np.floor(circle[0]+Rho*np.cos(theta))  ##注意是像素坐标系
        points_y=np.floor(circle[1]+Rho*np.sin(theta))
        is_circle=False
        area_left_up=0.0
        area_right_up=0.0
        area_left_down=0.0
        area_right_down=0.0
        area_all=0.0
        for i in range(0, np.shape(points_y)[0]):
            if points_x[i]>np.shape(img)[1] or points_x[i]<0 or points_y[i]<0 or points_y[i]>np.shape(img)[0]:
                return is_circle
            if points_x[i]<=circle[0]:  # left
                if points_y[i]<=circle[1]:  # up
                    area_left_up+=img[int(points_y[i])][int(points_x[i])]
                else:
                    area_left_down+=img[int(points_y[i])][int(points_x[i])]
            else:  # right
                if points_y[i]<=circle[1]:  # up
                    # print(int(points_x[i]))
                    # print(points_x[i])
                    area_right_up+=img[int(points_y[i])][int(points_x[i])]
                else:
                    # print(int(points_x[i]))
                    area_right_down+=img[int(points_y[i])][int(points_x[i])]

        area_left_up, area_left_down, area_right_up, area_right_down= (area_circle/4-area_left_up/25.5), area_circle/4-area_left_down/25.5, area_circle/4-area_right_up/25.5, area_circle/4-area_right_down/25.5
        area_left_up, area_left_down, area_right_up, area_right_down=1*(area_left_up>0)*area_left_up, 1*(area_left_down>0)*area_left_down, 1*(area_right_up>0)*area_right_up, 1*(area_right_down>0)*area_right_down
        #转换为黑色部分
        area_all=area_left_up+area_right_up+area_left_down+area_right_down


        area_mean=area_all/4
        area_diff=np.sqrt((np.power((area_left_down-area_mean),2)+np.power((area_left_up-area_mean),2)+np.power((area_right_down-area_mean),2)+np.power(
            (area_right_up-area_mean),2))/4)
        print("圆心坐标   黑色部分   左上 左下  右上  右下 差 圆面积")
        print((circle[0], circle[1], area_left_up, area_left_down, area_right_up, area_right_down,area_diff, area_circle))
        if area_all/area_circle>=0.1:
            if area_left_up>0.15*area_mean and area_left_down>0.15*area_mean and area_right_up>0.15*area_mean and area_right_down>0.15*area_mean:
                if area_diff<=5000:
                    is_circle=True
        print(is_circle)
        return is_circle

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
        self.src_f = cv.cvtColor(self.captured, cv.COLOR_BGR2GRAY)#灰度化
        self.src_f=cv.equalizeHist(self.src_f)
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
        cv.imwrite("gabor.jpg",self.gabor_img)
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

    def btnDeHole_Clicked(self):#形态学滤波
        strel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))  # 形态学操作的结构元素为直径为5的圆
        self.open_img=cv.morphologyEx(self.thresh_img, cv.MORPH_OPEN, strel1)#开运算京可能保留目标信息
        self.close_img = cv.morphologyEx(self.open_img, cv.MORPH_CLOSE, strel1)  # 目标为黑，闭运算能去除小的假目标

        rows, cols = self.close_img.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.close_img, cols, rows, bytesPerLine, QImage.Format_Indexed8))
        #cv.imwrite("binary.jpg",self.close_img)

        # self.imagetoShow = self.gabor_img
        # self.imageShow()

        # self.btnContour.setEnabled(True)
        self.labelImage.update()

    def btnHough_Clicked(self):#圆检测
        self.circle_img=self.close_img
        cv.imwrite("binary.jpg",self.circle_img)
        circles = cv.HoughCircles(self.circle_img, cv.HOUGH_GRADIENT, dp=1, minDist=300, param1=300, param2=5, minRadius=50, maxRadius=250)
        # 对数据进行四舍五入变为整数
        if not circles.flatten().sum():
            print("there's no circle detected,please change params ")
        circles = np.uint16(np.around(circles))
        print("done")
        for i in circles[0, :]:
            # 画出来圆的边界
            if self.circle_judge(i,self.circle_img) :

                cv.circle(self.circle_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 画出来圆心
                cv.circle(self.circle_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        rows, cols = self.circle_img.shape
        channels = 1
        bytesPerLine = channels * cols
        self.raw_pixmap = QtGui.QPixmap.fromImage(
            QImage(self.circle_img, cols, rows, bytesPerLine, QImage.Format_Indexed8))
        self.labelImage.update()


    def user_paint_event(self, event):
        painter = QPainter(self.labelImage)
        if not (self.raw_pixmap is None):
            painter.drawPixmap(0, 0, int(self.raw_width * self.scale_ratio),
                               int(self.raw_height * self.scale_ratio), self.raw_pixmap)


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    window=PyQtIpDemo()
    window.show()
    sys.exit(app.exec_())
