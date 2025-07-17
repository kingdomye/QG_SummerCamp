import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img = cv.imread("imgs/target.jpg")
# blur = cv.GaussianBlur(img, (5, 5), 0)
# cv.imshow('blur', blur)
# edges = cv.Canny(img, 100, 200)

# cv.imshow('base', img)
# cv.imshow('edges', edges)
cap = cv.VideoCapture(0)
while True:
    _, frame = cap.read()
    mask = cv.Canny(frame, 100, 200)
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


cv.destroyAllWindows()
