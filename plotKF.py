import cv2 
import numpy as np

from src.Filtering import KalmanFilter as kf

state = np.array([[350],[500],[0],[0],[0]])
scov = np.ones((5,5))*10
scale = 10

motor_speeds = [0,0]

def pdf(state,scov,map: np.ndarray):
    mu = state[:1]
    sig = scov[:2,:2]
    normdet = 2*np.pi*np.sqrt(np.linalg.det(sig))
    siginv = np.linalg.inv(sig)

    for i in range(0,map.shape[0]):
        for j in range(0,map.shape[1]):
            dvec = [[i],[j]]-mu
            pcov = dvec.transpose() @ siginv @ dvec
            map[i,j] = np.exp(0.5*pcov)/normdet
            #print(i,j,pcov,map[i,j])
    
    return map

while True:
    state,scov,spd,b,c = kf.get_position(state,scov,0.1,False,[0,0],motor_speeds)
    map = np.zeros((700,1000),dtype=np.uint8)
    #draw ellipse
    center = state[:2].transpose()[0].astype(int)
    map = cv2.circle(map,center,1,255,1)
    u, s, Vt = np.linalg.svd(scov)
    angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
    width, height,angvar,spvar,angdotvar = np.sqrt(s)
    width = int(width)
    height = int(height)
    map = cv2.ellipse(map,center,(width,height),angle,0,360,color=200,thickness=1)
    #draw direction
    print(angvar,spvar)
    ang = state[2][0]
    spdvect = scale*(spd+spvar)*np.array([np.cos(ang),np.sin(ang)])
    dirvect = np.add(center,spdvect).astype(np.int32)
    map = cv2.line(map,center,dirvect,200,1)
    sup = int(abs(spd+spvar*scale))
    sdw = int(abs(spd*scale))
    map = cv2.ellipse(map,center,(sup,sup),0,-angvar,angvar,color=200,thickness=1)
    map = cv2.ellipse(map,center,(sdw,sdw),0,-angvar,angvar,color=200,thickness=1)



    img = map
    cv2.imshow("map pdf",img)
    
    key = (cv2.waitKey(1) & 0xFF)

    if key == ord('q'):
        break
    if key == ord('w'):
        motor_speeds = np.add(motor_speeds,[1,1])
        print(motor_speeds)
    if key == ord('s'):
        motor_speeds = np.add(motor_speeds,[-1,-1])
        print(motor_speeds)
    if key == ord('a'):
        motor_speeds = np.add(motor_speeds,[1,-1])
        print(motor_speeds)
    if key == ord('d'):
        motor_speeds = np.add(motor_speeds,[-1,1])
        print(motor_speeds)
