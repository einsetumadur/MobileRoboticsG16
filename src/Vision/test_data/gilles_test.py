from src.Local_Nav import psymap
import cv2
import numpy as np

robot = [200,200,0.0]

obstacles = [[[0,85],[85,0]],[[300,0],[300,400],[0,300],[400,300]]]

while True:
    black = np.zeros((700,1000,3))
    black = psymap.hallucinate_map(robot,obstacles,black)
    cv2.imshow("test",black)
    robot[2] += 0.1
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        exit()

