from src.Local_Nav import psymap as pm
from src.Motion_Control import thymio as th
import numpy as np
import time
import math

#function that resets the robot's position and recalculates the path for the new position.
async def kidnapping(node, cap, REFRAME, MAP_SHAPE, VISUALIZE):
    await th.stop_motor(node)
    print("determination of new path")
    th.init(cap, REFRAME, MAP_SHAPE, VISUALIZE) #calculate new path

#If the robot did not manage to avoid the obstacle with the ANN and the obstacle is detected very close to the robot,
#then we stop the thymio and go into front_avoidance_mode. After a lot of test, we noticed that this is specially the case 
#when the obstacle is placed late in front of thymio and its surface is orthogonal to the robot's direction.
async def Front_obst_avoidance_seq(client,node,angle):
        print("critical obsatcle avoidance mode")
        await th.stop_motor(node)
        await client.sleep(1)
        await th.rotate(client,angle,100)
        await th.motorset(node, 100,100)
        await client.sleep(2.5)
        await th.rotate(client,-angle,100)

#the main local_navigation function
async def local_navigation(client,node,rob,obstacles):
    local_navigation_state=1
    proximity_values = await th.get_proximity_values(client)

    #This are the tuned weights for the left and the right motor.
    #The weights corresponds to how much should the motor respond to each proximity value measured by the sensors
    w_l = [50,  40, 20, -40, -50]
    w_r = [-50, -40, -20,  40,  50]

    # Scale factors for sensors and constant factor
    sensor_scale = 2000
    
    x = [0,0,0,0,0]

    #Considering also the global obstacles as obstacles in the local_navigation: 
    x_glob=pm.hallucinate_map(rob,obstacles)

    #normal avoidance case
    if local_navigation_state:
        
        for i in range(5):
            # Get and scale inputs
            x[i] = (proximity_values[i] +x_glob[i])// sensor_scale

        #y corresponds to the motor's command
        y = [100,100]    

        for i in range(len(x)):    
            # Compute outputs of neurons and set motor powers
            y[0] = y[0] + x[i] * w_l[i]
            y[1] = y[1] + x[i] * w_r[i]

    # Set motor powers
    await th.motorset(node,y[0],y[1])
    
    if(proximity_values[2]>3800):
        await Front_obst_avoidance_seq(client,node,np.pi/2)
            

    