import math
import numpy as np
import cv2
from src.Global_Nav import helpers_global as gb
from src.Vision import vision as vs


#Thymio control functions 
def motors(l_speed=500, r_speed=500):
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }

async def forward(node,motor_speed):
    await node.set_variables(motors(motor_speed,motor_speed))

async def move_forward(client, motor_speed, dist):
    if not dist<0.01:
        node = await client.wait_for_node()
        await node.set_variables(motors(motor_speed, motor_speed))
        # wait time to get theta 1.44 is the factor to correct
        time=dist*10/motor_speed*1.40
        await(client.sleep(time))
        # stop the robot
        await node.set_variables(motors(0, 0))


async def motorset(node,motor_speed_left,motor_speed_right):
    await node.set_variables(motors(motor_speed_left,motor_speed_right))

async def rotate(client,theta, motor_speed): #theta is in radians
    
    node = await client.wait_for_node()
    if theta>=0:
        direction_rot=1
    else: 
        direction_rot=-1
    await node.set_variables(motors(motor_speed*direction_rot, -motor_speed*direction_rot))
    # wait time to get theta 1.44 is the factor to correct
    time=(abs(theta))*100/motor_speed*1.40
    await(client.sleep(time))
    # stop the robot
    await node.set_variables(motors(0, 0))

async def stop_motor(node):
    await node.set_variables(motors(0,0))

async def get_proximity_values(client):
    # Wait for the Thymio node
    node = await client.wait_for_node()
    # Wait for the proximity sensor variables
    await node.wait_for_variables({"prox.horizontal"})
    # Get the proximity values : v: Stands for "variables" and is used to access the cached variable values.
    proximity_values = node.v.prox.horizontal
    # Return the value of the front proximity sensor (index 2)
    return proximity_values[0:5]




async def move_to_goal (client, x_est, goal, speed): 
    position_robot = [x_est[0],x_est[1]]
    angle_robot = x_est[2]
    pos_idx = convert_to_idx(position_robot,2)
    angle_goal = compute_angle(pos_idx, goal)
    print(angle_goal-angle_robot)

    await rotate(client, angle_robot-angle_goal, speed)
    await move_forward(client, 100, 4)

async def move_to_goal2(client,rob_pos_abs, goal, motor_speed):
    x = rob_pos_abs[0]
    y = rob_pos_abs[1]
    angle_rob = rob_pos_abs[2]
    absolut_angle_to_goal=math.atan2(y, x)

    x_disp = goal[0] - x
    y_disp = goal[1] - y

    dist_to_goal = math.sqrt(x_disp**2 + y_disp**2)
    theta = (absolut_angle_to_goal- angle_rob) % (2 * math.pi)
    print(dist_to_goal)
    print(theta)
    await rotate(client,theta,motor_speed)
    await move_forward(client, motor_speed, dist_to_goal)



def compute_angle(x1,x2 ):
    """Calculer l'angle entre la direction actuelle et la direction vers l'arrivée."""
    y = x2[1] - x1[1]
    x = x2[0] - x1[0]
    
    return math.atan2(y, x)


def convert_to_idx(position, size_cell):
    idx =[0,0]
    idx[0] = int(np.floor(position[0]/size_cell))
    idx[1] = int(np.floor(position[1]/size_cell))
    return idx



def init(cap, REFRAME, MAP_SHAPE, VISUALIZE): 
    # Get the path 

    if REFRAME:
        Tmap = vs.get_warp(cap,MAP_SHAPE,10,1)

    while True:
        ret,frame = cap.read()
        if ret:
        # maps capture to map
            if REFRAME:
            
                frame = cv2.warpPerspective(frame,Tmap,MAP_SHAPE)
            # maps BGR to HLS color space for simplicity
            HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        
    
            fmap = vs.get_grid_fixed_map(frame,(50,35),50, robrad=150)
        
            # find dest position
            bool_dest, dest = vs.get_destination(frame)
        
            if bool_dest:
                dest = dest/10.0
                dest[1] = 70-dest[1]
                dest = gb.convert_to_idx(dest,2)
                dest = tuple(dest)
        
                gotpos,robpos,orient, ppmx = vs.get_Robot_position_orientation(HLS,5)
                if gotpos : 
                    start=robpos
                    start[1]= 700 -start[1]
                    state_estimation_prev2 = np.array([[start[0]],[start[1]], [orient], [0],[0]])
                    start = start/10.0
                    print(state_estimation_prev2)
                    start = gb.convert_to_idx(start,2)
                    start = tuple(start)
                    path = gb.global_final(fmap,start,dest, "8N", VISUALIZE)
                    P_estimation_prev =  np.diag([100, 100, 0.75, 10, 0.75])
                    
                    break

    return path, state_estimation_prev2, P_estimation_prev
        