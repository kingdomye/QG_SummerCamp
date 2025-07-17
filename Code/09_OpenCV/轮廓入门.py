import cv2 as cv
import numpy as np

img = cv.imread("imgs/target.jpg")
cap = cv.VideoCapture(0)
while True:
    _, frame = cap.read()
    res = cv.Canny(frame, 100, 200)
    cv.imshow('res', res)

    # frame_array = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    
    ret, thresh = cv.threshold(res, 100, 200, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(res, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # (x, y), radius = cv.minEnclosingCircle(contour)
        # center = (int(x), int(y))
        # radius = int(radius)
        # cv.circle(frame, center, radius, (255, 0, 0), 3)

    cv.imshow('frame', res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# img_array = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(img_array, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv.imshow('img', img)

# cv.waitKey(0)
print(hierarchy)
cv.destroyAllWindows()
