import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter.fourcc(*'MJPG')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
if not cap.isOpened():
    print("No Camera")
    exit()

while True:
    ret, frame = cap.read()
    ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)

    frame = cv.flip(frame, 0)
    out.write(frame)
    cv.imshow('frame', frame)

    if not ret:
        print("cant receive frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
