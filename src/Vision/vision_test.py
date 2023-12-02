import cv2
import vision
import numpy as np

REFRAME = False
MAP_SHAPE = (1000,700)

#img = cv2.imread("./test_data/test_map.png")
#Tmap = vision.get_warp_image(img,(1000,700),10)

cap = cv2.VideoCapture(0)
if REFRAME:
    Tmap = vision.get_warp(cap,MAP_SHAPE,10,1)
dest = [0,0]
orient = 0
robpos = [0,0]
pxpcm = 10
while True:
    ret,frame = cap.read()
    if ret:
        # maps capture to map
        if REFRAME:
            frame = cv2.warpPerspective(frame,Tmap,MAP_SHAPE)
        # maps BGR to HLS color space for simplicity
        HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        vizu = vision.visualizer(HLS)
        # get fixed obstacle map
        fmap = vision.get_grid_fixed_map(frame,(100,70),50)
        # get fixed obstacle contours
        cont = vision.get_obstacles(frame,50,10)
        # find dest position
        dest = vision.get_destination(frame)
        # find robot
        gotpos,robpos,pxpcm,orient = vision.get_Robot_position_orientation(HLS,1,5)

        #visualization functions
        omap =vision.grid_fixedmap_visualizer(fmap,MAP_SHAPE)
        obsimg = cv2.merge([omap,omap,omap])
        #vizu = cv2.bitwise_or(vizu,obsimg)
        vizu = vision.draw_obstacles_poly(vizu,cont,(255,255,0),2)
        vizu = cv2.circle(vizu,dest,20,(50,25,100),4)
        vizu = cv2.addWeighted(vizu,0.5,frame,0.5,0)
        if gotpos:
            vizu = vision.paint_robot(vizu,(0,0,200),robpos,orient,pxpcm)
            print("pos:{},{:.2f} dest:{}".format(robpos,orient,dest),end='\r')
        cv2.imshow('img',vizu)
    
    else:
        print("lost camera.")
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()