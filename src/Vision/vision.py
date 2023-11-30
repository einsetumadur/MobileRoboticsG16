import cv2 
import numpy as np
import math

GREEN_HSL_MIN    = (106,43,44)
GREEN_HSL_MAX    = (143,256,256)
BLUE_HSL_MIN     = (141,37,73)
BLUE_HSL_MAX     = (172,256,256)
RED_UP_HSL_MIN   = (240,50,39)
RED_UP_HSL_MAX   = (255,256,256)
RED_DOWN_HSL_MIN = (0,50,40)
RED_DOWN_HSL_MAX = (10,256,256)

# Blob parameters = filter (byArea,byCirc,byCol,byConv,byInert)
#                   values (minArea,maxArea,minCirc,...,maxInert)
CORNER_BLOB_FILT    = (True,False,True,False,False)
CORNER_BLOB_VAL     = (200,4000,0.5,1,255,0.2,1,0.1,1)
ROBOT_BLOB_FILT     = (True,True,True,True,False)
ROBOT_BLOB_VAL      = (80,1000,0.8,1,255,0.7,1,0.1,1)
DEST_BLOB_FILT      = (True,True,True,False,False)
DEST_BLOB_VAL       = (100,10000,0.8,1,255,0.7,1,0.1,1)

def blob_param(filters,values):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = filters[0]
    params.filterByCircularity = filters[1]
    params.filterByColor = filters[2]
    params.filterByConvexity = filters[3]
    params.filterByInertia = filters[4]
    params.minArea = values[0]
    params.maxArea = values[1]
    params.minCircularity = values[2]
    params.maxCircularity = values[3]
    params.blobColor = values[4]
    params.minConvexity = values[5]
    params.maxConvexity = values[6]
    params.minInertiaRatio = values[7]
    params.maxInertiaRatio = values[8]
    return params

def blob_point_list(frame,filters,values):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = filters[0]
    params.filterByCircularity = filters[1]
    params.filterByColor = filters[2]
    params.filterByConvexity = filters[3]
    params.filterByInertia = filters[4]
    params.minArea = values[0]
    params.maxArea = values[1]
    params.minCircularity = values[2]
    params.maxCircularity = values[3]
    params.blobColor = values[4]
    params.minConvexity = values[5]
    params.maxConvexity = values[6]
    params.minInertiaRatio = values[7]
    params.maxInertiaRatio = values[8]
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)
    return len(keypoints),cv2.KeyPoint_convert(keypoints)

def dist(a,b):
    return np.sqrt((b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]))

def reorder_border(corners):
    center = np.mean(corners,0)
    ordered = np.copy(corners)
    rel = corners-center
    rel = (rel > 0).astype(np.uint8)
    if np.sum(rel) == 4:
        idx = 0
        for pt in rel:
            match list(pt):
                case (0,0):
                    ordered[0] = corners[idx]
                case [0,1]:
                    ordered[2] = corners[idx]
                case [1,0]:
                    ordered[1] = corners[idx]
                case [1,1]:
                    ordered[3] = corners[idx]
                case _:
                    print("error: invalid case {}".format(pt))
            idx += 1
        return True,ordered
    else:
        return False,ordered

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
            hslimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
            Blue = cv2.inRange(hslimg, BLUE_HSL_MIN, BLUE_HSL_MAX)
            Blue = cv2.morphologyEx(Blue, cv2.MORPH_CLOSE,kernel)
            ndot,corn = blob_point_list(Blue,CORNER_BLOB_FILT,CORNER_BLOB_VAL)
            corn = np.float32(corn)
            print("progress {} valid samples".format(ndot,smp),end='\r')
            if ndot == 4:
                valid,corn = reorder_border(corn)
                if valid:
                    corners[:,:,smp] = corn
                    smp += 1
    avgCorn = np.mean(corners,axis=2,dtype=np.float32)    
    print("Finished Warp Matrix computation.")       
    return cv2.getPerspectiveTransform(avgCorn,Sheetpts)

def get_warp_image(cimg,ROI,pad=0):
    kernel = np.ones((4,4),np.uint8)
    rw = ROI[0] - pad
    rh = ROI[1] - pad
    Sheetpts = np.float32([[pad,pad],[rw,pad],[pad,rh],[rw,rh]])
    print("Computing warp matrix")
    hslimg= cv2.cvtColor(cimg, cv2.COLOR_BGR2HLS_FULL)
    Blue = cv2.inRange(hslimg, BLUE_HSL_MIN, BLUE_HSL_MAX)
    Blue = cv2.morphologyEx(Blue, cv2.MORPH_CLOSE,kernel)
    ndot,corn = blob_point_list(Blue,CORNER_BLOB_FILT,CORNER_BLOB_VAL)
    corn = np.float32(corn)
    if ndot == 4:
        print("Success !")
        return cv2.getPerspectiveTransform(corn,Sheetpts)
    else:
        while True:
            cv2.imshow("corner",Blue)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        print("Error: {} corners found !".format(ndot),end='\r')
        return [[0,0],[0,0]]



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
    frame = cv2.line(frame,Ap,Bp,color=col,thickness=thick)
    frame = cv2.line(frame,Bp,Cp,color=col,thickness=thick)
    frame = cv2.line(frame,Cp,Dp,color=col,thickness=thick)
    frame = cv2.ellipse(frame,pos,(radius,radius),-ori*60.0,-fang,fang,color=col,thickness=thick)
    frame = cv2.line(frame,pos,front,color=col,thickness=thick)
    return frame

def visualizer(HLSframe):
  kernel = np.ones((5,5),np.uint8)
  Redup = cv2.inRange(HLSframe,RED_UP_HSL_MIN, RED_UP_HSL_MAX)
  Reddown = cv2.inRange(HLSframe,RED_DOWN_HSL_MIN, RED_DOWN_HSL_MAX)
  Red = cv2.bitwise_or(Redup,Reddown)
  Red = cv2.morphologyEx(Red, cv2.MORPH_CLOSE, kernel)
  Blue = cv2.inRange(HLSframe, (141,37,75), BLUE_HSL_MAX)
  Blue = cv2.morphologyEx(Blue, cv2.MORPH_CLOSE, kernel)
  Green = cv2.inRange(HLSframe, GREEN_HSL_MIN, GREEN_HSL_MAX)
  Green = cv2.morphologyEx(Green, cv2.MORPH_CLOSE, kernel)
  reddetector = cv2.SimpleBlobDetector_create(blob_param(ROBOT_BLOB_FILT,ROBOT_BLOB_VAL))
  bluedetector = cv2.SimpleBlobDetector_create(blob_param(CORNER_BLOB_FILT,CORNER_BLOB_VAL))
  greendetector = cv2.SimpleBlobDetector_create(blob_param(DEST_BLOB_FILT,DEST_BLOB_VAL))
  keypointsB = bluedetector.detect(Blue) 
  keypointsR = reddetector.detect(Red)
  keypointsG = greendetector.detect(Green)
  comp = cv2.merge([Blue,Green,Red])
  compd = cv2.drawKeypoints(comp, keypointsB, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  compd = cv2.drawKeypoints(compd, keypointsR, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  compd = cv2.drawKeypoints(compd, keypointsG, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return compd

def get_grid_fixed_map(frame,shape,tresh):
    kernel = np.ones((5,5),np.uint8)
    pxmap = cv2.inRange(frame,(0,0,0),(tresh,tresh,tresh))
    pxmap = cv2.morphologyEx(pxmap,cv2.MORPH_OPEN,kernel)
    temp = cv2.resize(pxmap, shape, interpolation=cv2.INTER_LINEAR)
    _,output = cv2.threshold(temp,10,1,type=cv2.THRESH_BINARY)
    return output

def grid_fixedmap_visualizer(fmap,shape):
    fmap = fmap*255
    return cv2.resize(fmap, shape, interpolation=cv2.INTER_NEAREST)

def get_obstacles(frame,tresh,eps):
    kernel = np.ones((5,5),np.uint8)
    pxmap = cv2.inRange(frame,(0,0,0),(tresh,tresh,tresh))
    pxmap = cv2.morphologyEx(pxmap,cv2.MORPH_OPEN,kernel)
    contp,hier =  cv2.findContours(pxmap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for cont in contp:
        cont = np.array(cont,dtype=np.float32).reshape((-1,2))
        scont = cv2.approxPolyDP(cont,eps,True)
        obstacles.append(scont)
    return obstacles

def draw_obstacles_poly(frame,obs_poly,color,thickness):
    for obs in obs_poly:
        obs = np.array(obs,np.int32).reshape(-1,2)
        frame = cv2.polylines(frame,[obs],True,color,thickness)
        for pts in obs:
            frame = cv2.circle(frame,pts,2*thickness,color,thickness)
    return frame
    
def get_destination(frame):
    kernel = np.ones((5,5),np.uint8)
    hslimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
    Green = cv2.inRange(hslimg, GREEN_HSL_MIN, GREEN_HSL_MAX)
    Green = cv2.morphologyEx(Green, cv2.MORPH_CLOSE,kernel)
    ndot,blobs = blob_point_list(Green,DEST_BLOB_FILT,DEST_BLOB_VAL)
    if ndot == 1:
        return np.int32(blobs[0])
    else:
        print("Warning: {} goals found".format(ndot),end='\r')
        return (0,0)


def get_Robot_position_orientation(hls_frame,blobfil,kernsize):

    kernel = np.ones((kernsize,kernsize),np.uint8)
    errpx = 2
    ratio = 4/6
    # 1 - filter for dots
    Redup = cv2.inRange(hls_frame,(240,60,75), (255,255,255))
    Reddown = cv2.inRange(hls_frame,(0,60,75), (10,255,255))
    Red = cv2.bitwise_or(Redup,Reddown)
    Red = cv2.morphologyEx(Red, cv2.MORPH_CLOSE, kernel)
    ndot,ptlist = blob_point_list(Red,ROBOT_BLOB_FILT,ROBOT_BLOB_VAL)

    # 3 - compute distances
    d_table = np.zeros((ndot,ndot))
    if (ndot >= 3):
        for p1 in range(0,ndot-1):
            for p2 in range(p1+1,ndot):
                d_table[p1,p2] = dist(ptlist[p1],ptlist[p2])
    else: 
        print("red blob nb {}".format(ndot))
        return False,[0,0],0,[0,0]
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
        print("wrong pattern")    
        return False,[0,0],0,[0,0]
    
