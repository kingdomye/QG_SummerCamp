import cv2 as cv
import numpy as np

# img = np.zeros((512, 512, 3), np.uint8)
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)

img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while True:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
