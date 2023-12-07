from src.Local_Nav import psymap as pm
from src.Motion_Control import thymio as th
import numpy as np
import time
import math


async def kidnapping(node, cap, REFRAME, MAP_SHAPE, VISUALIZE):
    await th.stop_motor(node)
    print("determination of new path")
    th.init(cap, REFRAME, MAP_SHAPE, VISUALIZE) #calculate new path
    #Attention, oublier pas de remettre le state estimate du filtering à 0 aubin


#Local navigation : Documentation.
#The requirements of local navigation is to react to obstacles not detected previusly by the camera.
#In our case, we 3D printed the local obstacles and in white so that they won't be detected by the camera.
#The main requirement is to avoid the obstacle. We do this by implemennting  

# def find_nearest_obstacle_direction(obstacles, center, orient):
#     center[0]=robot_x
#     center[1]=robot_y
#     # Calcule les distances entre le robot et chaque obstacle
#     distances = [math.sqrt((obstacle[0] - robot_x)**2 + (obstacle[1] - robot_y)**2) for obstacle in obstacles]

#     # Trouve l'index de l'obstacle le plus proche
#     nearest_obstacle_index = distances.index(min(distances))

#     # Calcule l'angle entre le robot et l'obstacle le plus proche
#     angle_to_obstacle = math.atan2(obstacles[nearest_obstacle_index][1] - robot_y, obstacles[nearest_obstacle_index][0] - robot_x)

#     # Calcule la différence d'angle entre l'angle du robot et l'angle vers l'obstacle
#     angle_difference = angle_to_obstacle - orient

#     # Détermine si l'obstacle est à gauche ou à droite du robot
#     if -math.pi/2 < angle_difference < math.pi/2:
#         print("Obstacle global closer on the right")
#         return -1
#     else:
#         print("Obstacle global closer on the left")
#         return 1

async def Front_obst_avoidance_seq(client,node,angle):
        print("critical obsatcle avoidance mode")
        await th.stop_motor(node)
        await client.sleep(1)
        await th.rotate(client,angle,100)
        await th.motorset(node, 100,100)
        await client.sleep(2.5)
        await th.rotate(client,-angle,100)

#The local navigation goal is to respond act 
async def local_navigation(client,node,rob,obstacles):
    local_navigation_state=1
    proximity_values = await th.get_proximity_values(client)
    state=1

    #This are the tuned weights for the left and the right motor.
    #The weights corresponds to how much should the motor respond to each proximity value measured by the sensors
    w_l = [40,  -40, 20, -40, -40]
    w_r = [-40, -40, 20,  20,  40]

    # Scale factors for sensors and constant factor
    sensor_scale = 2000
    
    x = [0,0,0,0,0]

    #Considering also the global obstacle as obstacles : 
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
    
    #considering the critical case where the left sensors tetect method 1
    #methode 1
    methode =0
    if methode==1:
        diff_left_right=abs(((x[0]+x[1])-(x[3]+x[4]))*sensor_scale)
        if(diff_left_right<2500 and sum(x)*sensor_scale>3500):
            if(x_glob[0]+x_glob[1]>x_glob[3]+x_glob[4]):
                await th.rotate(client,np.pi/2, 100)
                await th.motorset(node,100,100)
                time.sleep(2)
            else:
                await th.rotate(client,-np.pi/2, 100)
                await th.motorset(node,100,100)
                time.sleep(2)

            proximity_values = await th.get_proximity_values(client)
    #methode 2 : in the case where teh robot is very close and 
    #the obstacle surface is perfectly orthogonal to the robot
    if methode ==2:
        print("y[0-y[1]",abs(y[0]-y[1]),"x3",x[3])
        print(x[3])
        if(abs(y[0]-y[1])<200 and x[3]*sensor_scale>2500):#probably too low 
            if(x_glob[0]+x_glob[1]>x_glob[3]+x_glob[4]):
                await th.rotate(client,np.pi/2, 100)
                await th.motorset(node,100,100)
                time.sleep(2)
            else:
                await th.rotate(client,-np.pi/2, 100)
                await th.motorset(node,100,100)
                time.sleep(2)

    # Set motor powers
    await th.motorset(node,y[0],y[1])

async def local_navigation2(client,node,rob,obstacles):
    local_navigation_state=1
    proximity_values = await th.get_proximity_values(client)

    #This are the tuned weights for the left and the right motor.
    #The weights corresponds to how much should the motor respond to each proximity value measured by the sensors
    w_l = [50,  40, 20, -40, -50]
    w_r = [-50, -40, -20,  40,  50]

    # Scale factors for sensors and constant factor
    sensor_scale = 2000
    
    x = [0,0,0,0,0]

    #Considering also the global obstacle as obstacles : 
    x_glob=pm.hallucinate_map(rob,obstacles)

    #normal avoidance case
    if local_navigation_state:
        
        for i in range(5):
            # Get and scale inputs
            x[i] = (proximity_values[i] +x_glob[i])// sensor_scale
            #x[i] = (proximity_values[i] )// sensor_scale

        #y corresponds to the motor's command
        y = [100,100]    

        for i in range(len(x)):    
            # Compute outputs of neurons and set motor powers
            y[0] = y[0] + x[i] * w_l[i]
            y[1] = y[1] + x[i] * w_r[i]

    # Set motor powers
    await th.motorset(node,y[0],y[1])
    #considering the critical case where the left sensors tetect method 1
    #methode 1 : sensor du milieu detecte lobstacle trop proche : capteur à >4750 alors tourner à gauche/droite puis avancer 1s
#Front obstacle avoidance system.
#If the robot did not manage to avoid the obstacle with the ANN and the obstacle is detected very close to the robot,
#then we stop the thymio and go into front_avoidance_mod. After a lot of test, we noticed that this is specially the case 
#for small obstacles, when it's surface is orthogonal to the robots direction.

    if(proximity_values[2]>3800):
        index=2
        #sens=find_nearest_obstacle_direction(obstacles, center, orient)
        await Front_obst_avoidance_seq(client,node,np.pi/2)
        print("aaaaaaa")
            

    