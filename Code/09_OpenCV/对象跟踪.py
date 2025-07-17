import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description="The MeanShift algorithm")
parser.add_argument('--image', type=str, help='File path', default='imgs/target.jpg')
args = parser.parse_args()

cap = cv.VideoCapture(0)
ret, frame = cap.read()
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # meanShift
        # ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # x, y, w, h = track_window
        # img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        # cv.imshow('img2', img2)

        # camShift
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        pts = cv.boxPoints(ret)
        pts = np.int8(pts)
        img2 = cv.polylines(frame, [pts], True, (0, 0, 255), 2)
        cv.imshow('img2', img2)

        k = cv.waitKey(27) & 0xFF
        if k == 27:
            break
    else:
        break
