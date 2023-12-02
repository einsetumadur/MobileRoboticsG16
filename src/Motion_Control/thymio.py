import math
import numpy as np


#Thymio control functions 
def motors(l_speed=500, r_speed=500):
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }

async def forward(node,motor_speed):
    await node.set_variables(motors(motor_speed,motor_speed))

async def move_forward(client, motor_speed, dist):
    if not dist<0.001:
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
    if not (theta < 0.01 ):
        node = await client.wait_for_node()
        direction_rot=(theta>=0)-(theta<0)
        await node.set_variables(motors(motor_speed*direction_rot, -motor_speed*direction_rot))
        # wait time to get theta 1.44 is the factor to correct
        time=(theta)*100/motor_speed*1.40
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
    await move_forward(client, 100, 2)


def compute_angle(x1,x2 ):
    """Calculer l'angle entre la direction actuelle et la direction vers l'arrivée."""
    angle_rad = math.atan2(x2[1] - x1[1], x2[0] - x1[0])
    #angle_deg = math.degrees(angle_rad)
    return angle_rad


def convert_to_idx(position, size_cell):
    idx =[0,0]
    idx[0] = int(np.floor(position[0]/size_cell))
    idx[1] = int(np.floor(position[1]/size_cell))
    return idx