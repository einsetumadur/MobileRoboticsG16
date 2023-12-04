#from src.Local_Nav import psymap as pm
from src.Motion_Control import thymio as th
#async def local_navigation(rob,obstacles):
async def local_navigation(client,node):
    proximity_values = await th.get_proximity_values(client)
    state=1

    w_l = [40,  20, -20, -20, -40,  30, -10, 8, 0]
    
    w_r = [-40, -20, -20,  20,  40, -10, 30, 0, 8]

    # Scale factors for sensors and constant factor
    sensor_scale = 2000
    
    x = [0,0,0,0,0,0,0,0,0]
    
    #Considering also the global obstacle as obstacles : 
    #x_glob=pm.hallucinate_map(rob,obstacles,img=None)

    if state != 0:
        for i in range(5):
            # Get and scale inputs
            #x[i] = (proximity_values[i] +x_glob[i])// sensor_scale
            x[i] = (proximity_values[i]//sensor_scale)
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