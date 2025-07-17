import cv2 as cv
import numpy as np

img = cv.imread("imgs/target.jpg")
kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=1)

cv.imshow('img', img)
cv.imshow('erode', dilation)

cv.waitKey(0)
cv.destroyAllWindows()
