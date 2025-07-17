from __future__ import print_function

import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description="This program shows background")
parser.add_argument("--input", type=str, help='the path of video', default='vtest.avi')
parser.add_argument("--algo", type=str, help="Method:KNN/MOG2", default="KNN")
args = parser.parse_args()

if args.algo == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Unable to open:" + args.input)
    exit()
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
