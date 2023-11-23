#This function represents the local navigation of our project : 

#Important information : sensor detect the obstacle's accurately from 3cm to 13cm ; Over 14cm it doesn't detect them anymore.
#Goal : 
#The Robot will avoid the obstacle either on the right if obstacle detected on teh right side, either on the left.
#The obstacle is assumed as passed if the absolute position of the robot is back on the track computed by 
#A* (= finite number of absolute coordinates on the official path.)
#we will implement an uncertainty of the position of 2 meshs

#Local navigation True/False (Bolean) 


#Steps: 1.sensors values as input
#       2.create a local map : plot it for tests.
#       3.2 either put the coordinates into array of vision
#       3.3 or escape on our own the obstacle

def obstacles_pos_from_sensor_vals(sensor_vals):
    """
    Returns a list containing the position of the obstacles
    w.r.t the center of the Thymio robot. 
    :param sensor_vals: sensor values provided clockwise starting from the top left sensor.
    :return: numpy.array() that contains the position of the different obstacles
    """
    dist_to_sensor = [sensor_val_to_cm_dist(x) for x in sensor_vals]
    dx_from_sensor = [d*math.cos(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    dy_from_sensor = [d*math.sin(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    obstacles_pos = [[x[0]+dx, x[1]+dy] for (x,dx,dy) in zip(sensor_pos_from_center,dx_from_sensor,dy_from_sensor )]
    return np.array(abs.obstacles_pos)