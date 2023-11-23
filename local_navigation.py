#This function represents the local navigation of our project : 

#Important information : sensor detect the obstacle's accurately from 3cm to 13cm ; Over 14cm it doesn't detect them anymore.

#The goal of this function is only to avoid the obstacles. After avoiding the obstacles, we will again compute a new path with the global navigation.


def prox():
    global prox_ground_delta, motor_left_target, motor_right_target, BASICSPEED, GAIN
    diff = prox_ground_delta[1] - prox_ground_delta[0]
    motor_left_target = BASICSPEED - diff*GAIN
    motor_right_target = BASICSPEED + diff*GAIN

def timer0():
    global prox_ground_delta, prox_horizontal, motor_left_target, motor_right_target, state, obst, obstThrH, obstThrL, obstSpeedGain, speed0, speedGain 
    # acquisition from ground sensor for going toward the goal
    diffDelta = prox_ground_delta[1] - prox_ground_delta[0]

    # acquisition from the proximity sensors to detect obstacles
    obst = [prox_horizontal[0], prox_horizontal[4]]
    
    # tdmclient does not support yet multiple and/or in if statements:
    if state == 0: 
        # switch from goal tracking to obst avoidance if obstacle detected
        if (obst[0] > obstThrH):
            state = 1
        elif (obst[1] > obstThrH):
            state = 1
    elif state == 1:
        if obst[0] < obstThrL:
            if obst[1] < obstThrL : 
                # switch from obst avoidance to goal tracking if obstacle got unseen
                state = 0
    if  state == 0 :
        # goal tracking: turn toward the goal
        leds_top = [0,0,0]
        motor_left_target = speed0 - speedGain * diffDelta
        motor_right_target = speed0 + speedGain * diffDelta
    else:
        leds_top = [30,30,30]
        # obstacle avoidance: accelerate wheel near obstacle
        motor_left_target = speed0 + obstSpeedGain * (obst[0] // 100)
        motor_right_target = speed0 + obstSpeedGain * (obst[1] // 100)
