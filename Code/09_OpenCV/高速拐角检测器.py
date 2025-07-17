import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread("imgs/target1.jpeg")
# fast = cv.FastFeatureDetector.create()
# kp = fast.detect(img, None)
#
# img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
# cv.imshow('img2', img2)
#
# fast.setNonmaxSuppression(False)
# kp = fast.detect(img, None)
# img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
# cv.imshow('img3', img3)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# BRIEF
# img = cv.imread("imgs/target.jpg")
# star = cv.xfeatures2d.StarDetector.create()
# brief = cv.xfeatures2d.BriefDescriptorExtractor.create()
# kp = star.detect(img, None)
# kp, des = brief.compute(img, kp)
# print(brief.descriptorSize(), des.shape)

# ORB
img = cv.imread("imgs/face.jpg", 0)
orb = cv.ORB.create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2)
plt.show()
