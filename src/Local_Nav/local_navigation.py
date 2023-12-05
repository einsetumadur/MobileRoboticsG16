from src.Local_Nav import psymap as pm
from src.Motion_Control import thymio as th
import numpy as np
import time

#Local navigation : Documentation.
#The requirements of local navigation is to react to obstacles not detected previusly by the camera.
#In our case, we 3D printed the local obstacles and in white so that they won't be detected by the camera.
#The main requirement is to avoid the obstacle. We do this by implemennting  

#The local navigation goal is to respond act 
async def local_navigation(client,node,rob,obstacles):
    local_navigation_state=1
    proximity_values = await th.get_proximity_values(client)
    state=1

    #This are the tuned weights for the left and the right motor.
    #The weights corresponds to how much should the motor respond to each proximity value measured by the sensors
    w_l = [40,  20, 20, -40, -40]
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