from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        faceROI = frame_gray[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            eye_rect = (x + x2, y + y2, w2, h2)
            frame = cv.rectangle(frame, (eye_rect[0], eye_rect[1]),
                                 (eye_rect[0] + eye_rect[2], eye_rect[1] + eye_rect[3]), (255, 0, 0), 2)

    cv.imshow('Capture', frame)


parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument('--face_cascade', help='face cascade path', default='opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!)No captured frame -- Break!')
        break
    detectAndDisplay(frame)

    if cv.waitKey(10) & 0xFF == 27:
        break
