import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("imgs/target.jpg")
# cv.imshow('img', img)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()

# 直方图均衡
# equ = cv.equalizeHist(img)
# res = np.hstack((img, equ))
# cv.imshow('res', res)

# 直方图均衡优化
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cll = clahe.apply(img)
# cv.imshow('img', img)
# cv.imshow('res', cll)

# 二维直方图
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# plt.imshow(hist, interpolation='nearest')
# plt.show()

cap = cv.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(frame)
    res = np.hstack((frame, equ))
    cv.imshow('res', res)

    if cv.waitKey(5) & 0xFF == 27:
        break

# cv.waitKey(0)
cv.destroyAllWindows()
