# Gilles Regamey 2023

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import vision

#vision.list_ports()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    concat = np.hstack((vision.ColorRatioSel(frame,(0,0,1),(0.4,0.4,0.35))[0:200,0:300],
                        vision.ColorRatioSel(frame,(0,1,0),(0.5,0.35,0.6))[0:200,0:300],
                        vision.ColorRatioSel(frame,(1,0,0),(0.4,0.35,0.35))[0:200,0:300]))

    cv.imshow('frame, q to quit', concat)
    #cv.imshow('frame, q to quit', vision.ColorThreshSelect(frame,(0,0,1),(100,100,150)))

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()