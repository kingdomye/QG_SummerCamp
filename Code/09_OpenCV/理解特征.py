import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 哈里斯角检测
# img = cv.imread("imgs/sudok.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 2, 3, 0.04)
# img[dst > 0.01*dst.max()] = [255, 0, 0]
# cv.imshow('dst', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Shi-tomas拐角检测器
# img = cv.imread("imgs/target1.jpeg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
# corners = np.int8(corners)
# for i in corners:
#     x, y = i.ravel()
#     cv.circle(img, (x, y), 3, 255, -1)
# plt.imshow(img)
# plt.show()

# SIFT
img = cv.imread("imgs/target1.jpeg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT.create()
kp, des = sift.detectAndCompute(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
