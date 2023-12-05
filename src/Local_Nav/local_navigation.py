from src.Local_Nav import psymap as pm
from src.Motion_Control import thymio as th

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
    else: 
        # In case we would like to stop the robot
        y = [0,0] 
    
    # Set motor powers
    await th.motorset(node,y[0],y[1])