import cv2 as cv
import numpy as np

def ColorThreshSelect(image,colcode,tresh):

    (B,G,R) = cv.split(image)
    ret,Bt = cv.threshold(B,tresh[0],10,cv.THRESH_BINARY)
    ret,Gt = cv.threshold(G,tresh[1],15,cv.THRESH_BINARY)
    ret,Rt = cv.threshold(R,tresh[2],20,cv.THRESH_BINARY)
    compo = cv.add(Bt,Gt)
    compo = cv.add(compo,Rt)

    match colcode:
        case (0,0,0):
            return cv.inRange(compo,0,5)
        case (1,0,0):
            return cv.inRange(compo,5,12)
        case (0,1,0):
            return cv.inRange(compo,12,17)
        case (0,0,1):
            return cv.inRange(compo,17,22)
        case (1,1,0):
            return cv.inRange(compo,22,27)
        case (1,0,1):
            return cv.inRange(compo,27,32)
        case (0,1,1):
            return cv.inRange(compo,32,40)
        case (1,1,1):
            return cv.inRange(compo,40,255)
        case _:
            return  compo
        
def ColoradaptSelect(image,colcode,blksize):

    (B,G,R) = cv.split(image)
    Bt = cv.adaptiveThreshold(B,10,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blksize,0)
    Gt = cv.adaptiveThreshold(G,15,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blksize,0)
    Rt = cv.adaptiveThreshold(R,20,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blksize,0)
    compo = cv.add(Bt,Gt)
    compo = cv.add(compo,Rt)

    match colcode:
        case (0,0,0):
            return cv.inRange(compo,0,5)
        case (1,0,0):
            return cv.inRange(compo,5,12)
        case (0,1,0):
            return cv.inRange(compo,12,17)
        case (0,0,1):
            return cv.inRange(compo,17,22)
        case (1,1,0):
            return cv.inRange(compo,22,27)
        case (1,0,1):
            return cv.inRange(compo,27,32)
        case (0,1,1):
            return cv.inRange(compo,32,40)
        case (1,1,1):
            return cv.inRange(compo,40,255)
        case _:
            return  compo

def ColorRatioSel(image,colcode,ratio):
    (B,G,R) = cv.split(image)
    total = cv.add(B,G,dtype=2)
    total = cv.add(total,R,dtype=2)
    Br = cv.divide(B,total,dtype=5)
    Gr = cv.divide(G,total,dtype=5)
    Rr = cv.divide(R,total,dtype=5)

    match colcode:
        case (0,0,0):
            ret,Bt = cv.threshold(Br,ratio[0],255,cv.THRESH_BINARY_INV)
            ret,Gt = cv.threshold(Gr,ratio[1],255,cv.THRESH_BINARY_INV)
            ret,Rt = cv.threshold(Rr,ratio[2],255,cv.THRESH_BINARY_INV)
            return cv.bitwise_and(Rt,cv.bitwise_and(Gt,Bt))
        case (1,0,0):
            ret,Bt = cv.threshold(Br,ratio[0],255,cv.THRESH_BINARY)
            ret,Gt = cv.threshold(Gr,ratio[1],255,cv.THRESH_BINARY_INV)
            ret,Rt = cv.threshold(Rr,ratio[2],255,cv.THRESH_BINARY_INV)
            return cv.bitwise_and(Rt,cv.bitwise_and(Gt,Bt))
        case (0,1,0):
            ret,Bt = cv.threshold(Br,ratio[0],255,cv.THRESH_BINARY_INV)
            ret,Gt = cv.threshold(Gr,ratio[1],255,cv.THRESH_BINARY)
            ret,Rt = cv.threshold(Rr,ratio[2],255,cv.THRESH_BINARY_INV)
            return cv.bitwise_and(Rt,cv.bitwise_and(Gt,Bt))
        case (0,0,1):
            ret,Bt = cv.threshold(Br,ratio[0],255,cv.THRESH_BINARY_INV)
            ret,Gt = cv.threshold(Gr,ratio[1],255,cv.THRESH_BINARY_INV)
            ret,Rt = cv.threshold(Rr,ratio[2],255,cv.THRESH_BINARY)
            return cv.bitwise_and(Rt,cv.bitwise_and(Gt,Bt))
        case (1,1,1):
            ret,Bt = cv.threshold(Br,ratio[0],255,cv.THRESH_BINARY)
            ret,Gt = cv.threshold(Gr,ratio[1],255,cv.THRESH_BINARY)
            ret,Rt = cv.threshold(Rr,ratio[2],255,cv.THRESH_BINARY)
            return cv.bitwise_and(Rt,cv.bitwise_and(Gt,Bt))
        case _:
            print("error : "+colcode+" not implemented.")
            return image