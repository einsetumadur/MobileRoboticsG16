#function to hallucinate fixed obstacle map to ir sensors 
import numpy as np
import cv2 
#sensors points for prox horizontal in mm relative to robot center
SENSE_PROX_LIST = [[-51,61],[-27,75],[0,80],[27,75],[51,61],[-32,-30],[32,-30]]
#sensor range expected to be 0-50mm
SENSE_RANGE_LIST = [[-32,38],[-17,47],[0,50],[17,47],[32,38],[0,-50],[0,-50]]
OVER_RANGE = 100
PROX_MAX = 5000
PROX_MIN = 2800

def mm_to_sense(distmm):
    if distmm >= OVER_RANGE:
        return 0
    else:
        return int(PROX_MAX-distmm*(2200/50))

def sensor_to_line_val(sensor,range,l1, l2):
    sx,sy = sensor
    rx,ry = range
    lx1,ly1 = l1
    lx2,ly2 = l2
    denom = ((ly2-ly1)*(rx-sx)) - ((lx2-lx1)*(ry-sy))
    if denom == 0: # parallel
        return None,None,None
    ua = ((lx2-lx1)*(sy-ly1) - (ly2-ly1)*(sx-lx1)) / denom
    if ua < 0 or ua > 1: # out of range
        return None,None,None
    ub = ((rx-sx)*(sy-ly1) - (ry-sy)*(sx-lx1)) / denom
    if ub < 0 or ub > 1: # out of range
        return None,None,None
    intx = sx + ua * (rx-sx)
    inty = sy + ua * (ry-sy)
    dist = np.sqrt((sx-intx)*(sx-intx) + (sy-inty)*(sy-inty))
    return dist,intx,inty

def hallucinate_map(rob,obstacles,img=None):

    rx,ry,ra = rob
    cos = np.cos(ra)
    sin = np.sin(ra)
    rotMat = np.array([[sin,cos],[-cos,sin]])

    minproxdist = [OVER_RANGE,OVER_RANGE,OVER_RANGE,OVER_RANGE,OVER_RANGE,OVER_RANGE,OVER_RANGE]
    intersectP = np.zeros((7,2))

    for obstacle in obstacles:
        for obidx in range(1,len(obstacle)):
            for sensor in range(0,7):
                sensepoint = (rotMat @ SENSE_PROX_LIST[sensor] + [rx,ry])
                rangepoint = (rotMat @ np.add(SENSE_PROX_LIST[sensor],SENSE_RANGE_LIST[sensor]) + [rx,ry])
                if img is not None:
                    img = cv2.line(img,sensepoint.astype(np.int32),rangepoint.astype(np.int32),(100,100,100),1)
                segp1 = obstacle[obidx-1][0,:].tolist()
                segp2 = obstacle[obidx][0,:].tolist()
                print(sensepoint,rangepoint)
                dist,ix,iy = sensor_to_line_val(sensepoint,rangepoint,segp1,segp2)
                if (dist is not None) and (minproxdist[sensor] > dist):
                    if img is not None:
                        img = cv2.circle(img,np.array([ix,iy],dtype=np.int32),5,(0,255,0),3)
                    minproxdist[sensor] = dist
                    intersectP[sensor,:] = [ix,iy]

    hsenseval = minproxdist
    for i in range(0,7):
        hsenseval[i] = mm_to_sense(minproxdist[i])

    if img is not None:
        img = cv2.putText(img,str(hsenseval),(10,600),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        return img

    return hsenseval