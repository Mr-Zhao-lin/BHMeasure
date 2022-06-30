import cv2
import numpy as np

smarties = cv2.imread(r"jpg\2.jpg")

gray_img = cv2.cvtColor(smarties, cv2.COLOR_BGR2GRAY)
#gray_img=cv2.resize(gray_img,[int(1280/4),int(1024/4)])
# 进行中值滤波
img = cv2.medianBlur(gray_img, 5)
#ret,img=cv2.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv2.imshow("median",img)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=500, param1=240, param2=90, minRadius=60, maxRadius=500)
# 对数据进行四舍五入变为整数
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # 画出来圆的边界
    cv2.circle(gray_img, (i[0], i[1]), i[2], (0, 0, 0), 2)
    # 画出来圆心
    cv2.circle(gray_img, (i[0], i[1]), 2, (0, 255, 255), 3)
cv2.imshow("Circle", gray_img)
cv2.waitKey()
cv2.destroyAllWindows()