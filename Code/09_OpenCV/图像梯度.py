import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("imgs/target.jpg", 0)
laplacian = cv.Laplacian(img, cv.CV_64F)

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

cv.imshow('img', img)
cv.imshow('lap', laplacian)
cv.waitKey(0)
cv.destroyAllWindows()
