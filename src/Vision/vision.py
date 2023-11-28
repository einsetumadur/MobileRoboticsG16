import cv2 as cv
import numpy as np
import math

def printinfo(x,name):
    print("### {} ###".format(name))
    print(type(x))
    print(x)
    print(np.shape(x))

def blob_point_list_area(frame,minarea,maxarea,mincirc,minconv):
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = minarea
    params.maxArea = maxarea
    params.filterByCircularity = True
    params.minCircularity = mincirc
    params.filterByColor = True
    params.blobColor = 255
    params.filterByConvexity = False
    params.minConvexity = minconv
    params.maxConvexity = 1
    params.filterByInertia = False
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)
    return len(keypoints),cv.KeyPoint_convert(keypoints)

def dist(a,b):
    return np.sqrt((b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]))

def get_cam_distortion(cap):
    kernel = np.ones((4,4),np.uint8)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rw = 100
    rh = 70
    realcorn = [[0,0,0],[rw,rh,0],[0,rh,0],[rw,0,0]]
    objpts = [] 
    imgpts = [] 
    validframes = 0
    print("calibrating:")
    while validframes < 11:
        #print("validframes {}/10".format(validframes), end="\r")
        ret, img = cap.read()
        HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)
        Blue = cv.inRange(HLS, (141,37,137), (172,255,255))
        Blue = cv.morphologyEx(Blue, cv.MORPH_CLOSE,kernel)
        if ret:
            ndot,corners = blob_point_list_area(Blue,200,400,0.5,0.9)
            if ndot == 4:
                print(np.array(corners))
                print(realcorn)
                objpts.append(realcorn)
                imgpts.append(corners)
                validframes += 1
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpts,imgpts,Blue.shape[::-1],None,None)
    print(mtx)

def get_warp(cap,ROI,pad=0,samples=1):
    kernel = np.ones((4,4),np.uint8)
    rw = ROI[0] - pad
    rh = ROI[1] - pad
    Sheetpts = np.float32([[pad,pad],[rw,pad],[pad,rh],[rw,rh]])
    print("Computing warp matrix")
    smp = 0
    corners = np.zeros((4,2,samples))
    while smp < samples:
        ret, frame = cap.read()
        if ret:
            HLS = cv.cvtColor(frame, cv.COLOR_BGR2HLS_FULL)
            Blue = cv.inRange(HLS, (141,37,137), (172,255,255))
            Blue = cv.morphologyEx(Blue, cv.MORPH_CLOSE,kernel)
            ndot,corn = blob_point_list_area(Blue,200,400,0.5,0.9)
            corn = np.float32(corn)
            print("progress {}% ".format(ndot,100*smp/samples),end='\r')
            if ndot == 4:
                corners[:,:,smp] = corn
                smp += 1
    avgCorn = np.mean(corners,axis=2,dtype=np.float32)    
    print("Finished Warp Matrix computation.")       
    return cv.getPerspectiveTransform(avgCorn,Sheetpts)




def paint_robot(frame,col,pos,ori,scal):
    thick = 2
    fang = 42
    ori = -ori
    cos = np.cos(ori)
    sin = np.sin(ori)
    scal = int(scal)
    radius = 8*scal
    upcorn = 6.3*scal
    width = (11*scal)/2
    back = 3*scal
    rotMat = np.array([[sin, -cos],[cos, sin]])
    Ap = (rotMat @ [-width,-upcorn] + pos).astype(int)
    Bp = (rotMat @ [-width,back] + pos).astype(int)
    Cp = (rotMat @ [width,back] + pos).astype(int)
    Dp = (rotMat @ [width,-upcorn] + pos).astype(int)
    front = (rotMat @ [0,-radius] + pos).astype(int)
    #print("A{} B{} C{} D{} front{}".format(Ap,Bp,Cp,Dp,front))
    frame = cv.line(frame,Ap,Bp,color=col,thickness=thick)
    frame = cv.line(frame,Bp,Cp,color=col,thickness=thick)
    frame = cv.line(frame,Cp,Dp,color=col,thickness=thick)
    frame = cv.ellipse(frame,pos,(radius,radius),-ori*60.0,-fang,fang,color=col,thickness=thick)
    frame = cv.line(frame,pos,front,color=col,thickness=thick)
    return frame

def visualizer(HLSframe):
  kernel = np.ones((5,5),np.uint8)
  Redup = cv.inRange(HLSframe,(240,75,75), (255,255,255))
  Reddown = cv.inRange(HLSframe,(0,75,75), (10,255,255))
  Red = cv.bitwise_or(Redup,Reddown)
  Red = cv.morphologyEx(Red, cv.MORPH_CLOSE, kernel)
  Blue = cv.inRange(HLSframe, (141,37,75), (172,255,255))
  Blue = cv.morphologyEx(Blue, cv.MORPH_CLOSE, kernel)
  Green = cv.inRange(HLSframe, (98,0,70), (116,255,255))
  Green = cv.morphologyEx(Green, cv.MORPH_CLOSE, kernel)
  params = cv.SimpleBlobDetector_Params()
  params.filterByArea = True
  params.minArea = 20
  params.maxArea = 500
  params.filterByCircularity = False
  params.filterByColor = True
  params.blobColor = 255
  params.filterByConvexity = True
  params.minConvexity = 0.9
  params.maxConvexity = 1
  params.filterByInertia = False
  reddetector = cv.SimpleBlobDetector_create(params)
  params.filterByConvexity = True
  params.minConvexity = 0.9
  params.maxConvexity = 1
  bluedetector = cv.SimpleBlobDetector_create(params)
  keypointsB = bluedetector.detect(Blue) 
  keypointsR = reddetector.detect(Red)
  comp = cv.merge([Blue,Green,Red])
  compd = cv.drawKeypoints(comp, keypointsB, np.array([]), (255,255,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  compd = cv.drawKeypoints(compd, keypointsR, np.array([]), (255,255,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return compd

def get_grid_fixed_map(frame,shape,tresh):
    kernel = np.ones((5,5),np.uint8)
    pxmap = cv.inRange(frame,(0,0,0),(tresh,tresh,tresh))
    pxmap = cv.morphologyEx(pxmap,cv.MORPH_OPEN,kernel)
    temp = cv.resize(pxmap, shape, interpolation=cv.INTER_LINEAR)
    _,output = cv.threshold(temp,10,1,type=cv.THRESH_BINARY)
    return output

def grid_fixedmap_visualizer(fmap,shape):
    fmap = fmap*255
    return cv.resize(fmap, shape, interpolation=cv.INTER_NEAREST)

def get_obstacles(frame,tresh):
    kernel = np.ones((5,5),np.uint8)
    pxmap = cv.inRange(frame,(0,0,0),(tresh,tresh,tresh))
    pxmap = cv.morphologyEx(pxmap,cv.MORPH_OPEN,kernel)
    

def get_Robot_position_orientation(hls_frame,blobfil,kernsize):
    ###
    # input : frame  - opencv image (hls format)
    #
    ###
    kernel = np.ones((kernsize,kernsize),np.uint8)
    errpx = 2
    ratio = 4/6
    # 1 - filter for dots
    Redup = cv.inRange(hls_frame,(240,60,75), (255,255,255))
    Reddown = cv.inRange(hls_frame,(0,60,75), (10,255,255))
    Red = cv.bitwise_or(Redup,Reddown)
    Red = cv.morphologyEx(Red, cv.MORPH_CLOSE, kernel)
    ndot,ptlist = blob_point_list_area(Red,400,600,0.7,0.5)

    # 3 - compute distances
    d_table = np.zeros((ndot,ndot))
    if (ndot >= 3):
        for p1 in range(0,ndot-1):
            for p2 in range(p1+1,ndot):
                d_table[p1,p2] = dist(ptlist[p1],ptlist[p2])
    else: 
        return False,(0,0),0,(0,0)
    dmax = np.max(d_table)
    longidx = np.int8(abs(dmax - d_table) < errpx)
    shortidx = np.int8(abs(dmax*ratio - d_table) < errpx)
    #print("long:{} short:{}".format(np.sum(longidx),np.sum(shortidx)))
    if(np.sum(shortidx) == 1 and np.sum(longidx) == 2):
    # 4 - compute pos and orientation
        pairAB = np.where(shortidx == 1)
        Cidx = np.where((np.sum(longidx+np.transpose(longidx),1))==2)
        center = np.array([int((ptlist[pairAB[0][0]][0] + ptlist[pairAB[1][0]][0])/2),
                           int((ptlist[pairAB[0][0]][1] + ptlist[pairAB[1][0]][1])/2)])
        scale = d_table[pairAB]/4
        ptC = np.array([int(ptlist[Cidx[0][0]][0]),int(ptlist[Cidx[0][0]][1])])
        dirvect = ptC - center
        orient = np.arctan2(dirvect[1],dirvect[0])
        return True,center,scale,orient
    else:
        #print("wrong shape")    
        return False,(0,0),0,(0,0)
    
