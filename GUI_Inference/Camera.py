import numpy as np  # type: ignore
import cv2 as cv    # type: ignore

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (320, 240))

if not cap.isOpened():
    print("No camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    out.write(frame)
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
