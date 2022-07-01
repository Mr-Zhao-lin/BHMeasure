def circle_judge(circle,img):
    circle_r=circle[2]
    area_circle=np.pi*circle_r
    num_points=round(area_circle/10) #使用散点抽样来估计面积
    Rho=0#圆得极坐标参数1
    theta=0;#极坐标参数之角度
    Rho=np.sqrt(np.random.uniform(0, circle_r, num_points))*np.sqrt(circle_r)
    theta=np.random.uniform(0,2*np.pi,num_points)
    points_x=np.round(circle[0]+Rho*np.cos(theta))##注意是像素坐标系
    points_y=np.round(circle[1]+Rho*np.sin(theta))
    plt.scatter(points_x,points_y)
    plt.show()
    print(np.max(points_x))
    is_circle=False
    area_left_up=0.0
    area_right_up=0.0
    area_left_down=0.0
    area_right_down=0.0
    area_all=0.0
    for i in range(0,np.shape(points_y)[0]):
        if points_x[i]>np.shape(img)[0] or points_x[i]<0 or points_y[i]<0 or points_y[i]>np.shape(img)[1]:
            return is_circle
        if points_x[i]<=circle[0]:#left
            if points_y[i]<=circle[1]:#up
                area_left_up+=255#img[int(points_x[i])][int(points_y[i])]
            else:
                area_left_down+=255#img[int(points_x[i])][int(points_y[i])]
        else:#right
            if points_y[i]<=circle[1]:#up
                #print(int(points_x[i]))
               # print(points_x[i])
                area_right_up+=255#img[int(points_x[i])][int(points_y[i])]
            else:
                #print(int(points_x[i]))
                area_right_down+=255#img[int(points_x[i])][int(points_y[i])]
    area_left_up,area_left_down,area_right_up,area_right_down=area_left_up/25.5,area_left_down/25.5,area_right_up/25.5,area_right_down/25.5
    area_all=area_left_up+area_right_up+area_left_down+area_right_down
    print("圆心坐标   白色部分   左上 左下  右上  右下  圆面积")
    print((circle[0], circle[1], area_left_up, area_left_down, area_right_up, area_right_down, area_circle))
    #实际上是白色部分面积
    area_mean=area_all/4
    if area_all/area_circle<=2:
        if (abs(area_left_down-area_mean)+abs(area_left_up-area_mean)+abs(area_right_down-area_mean)+abs(area_right_up-area_mean))/4<=(1*area_mean):
            is_circle=True
    return area_all


import cv2
import numpy as np
import matplotlib.pyplot as plt

smarties = cv2.imread(r"binary.jpg")

gray_img = cv2.cvtColor(smarties, cv2.COLOR_BGR2GRAY)
#hist=cv2.calcHist(gray_img, [0], None, [256], [0, 256])

#cv2.imshow("hist",hist)
#直方图均衡化
gray_img=cv2.equalizeHist(gray_img)


#gray_img=cv2.resize(gray_img,[int(1280/4),int(1024/4)])
# 进行中值滤波
img = cv2.medianBlur(gray_img, 5)

#circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=300, param2=5, minRadius=50, maxRadius=500)
circles=np.array([500,500,200])
# 对数据进行四舍五入变为整数
circles = np.uint16(np.around(circles))
#剔除误检测得圆
#条件一：圆内黑色部分需要占到圆面积得5%以上
#条件二：圆内黑色部分需要相对对称

circle_judge(circles,img)


