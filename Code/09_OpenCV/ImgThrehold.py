import cv2 as cv
import numpy as np

img = cv.imread("imgs/target.jpg")
# kernel = np.ones((30, 30), np.float32) / 900
# dst = cv.filter2D(img, -1, kernel)
# cv.imshow('img', img)
# cv.imshow('dst', dst)

# cv.waitKey(0)
# cv.destroyAllWindows()

# blur = cv.blur(img, (5, 5))
blur = cv.bilateralFilter(img, 91, 75, 75)
cv.imshow('img', img)
cv.imshow('dst', blur)

cv.waitKey(0)
cv.destroyAllWindows()
