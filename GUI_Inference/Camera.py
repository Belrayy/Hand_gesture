import numpy as np # type: ignore
import cv2 as cv # type: ignore

cap=cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (320,  240))

if not cap.isOpened():
    print("No camera")
    exit()
while True:
    frame = cap.read()
    ret= cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
    ret= cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)
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