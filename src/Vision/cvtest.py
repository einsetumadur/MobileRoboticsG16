import cv2
import vision
import numpy as np

cap = cv2.VideoCapture(0)

#vision.get_cam_distortion(cap)

Tmap = vision.get_warp(cap,(1000,700),20,10)

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.warpPerspective(frame,Tmap,(1000,700))
        HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        fmap = vision.get_grid_fixed_map(frame,(100,70),50)
        omap =vision.grid_fixedmap_visualizer(fmap,(1000,700))
        obsimg = cv2.merge([omap,omap,omap])
        mixed = cv2.addWeighted(frame,0.5,obsimg,0.5,0)
        cv2.imshow('img',mixed)
    
    else:
        print("lost camera.")
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()