#Functions : 

#1: not used
speed0 = 100       # nominal speed
speedGain = 2      # gain used with ground gradient
obstThrL = 10      # low obstacle threshold to switch state 1->0
obstThrH = 20      # high obstacle threshold to switch state 0->1
obstSpeedGain = 5  # /100 (actual gain: 5/100=0.05)

state = 1          # 0=gradient, 1=obstacle avoidance
obst = [0,0]       # measurements from left and right prox sensors
angle=-10 #c'est pour les tests
#timer_period[0] = 10   # 10ms sampling time


async def local_navigation2():
    # acquisition from ground sensor for going toward the goal
    if (angle>0):
        obstacle_state="obstacle_passed"
    # acquisition from the proximity sensors to detect obstacles
    
    obst = await get_proximity_values()
    # tdmclient does not support yet multiple and/or in if statements:
    if obstacle_state=="obstacle_passed": 
        # switch from goal tracking to obst avoidance if obstacle detected
        if (obst[0] > obstThrH):
            obstacle_state = "obstacle_detected"
        elif (obst[5] > obstThrH):
            obstacle_state = "obstacle_detected"
    elif obstacle_state == "obstacle_detected":
        if obst[0] < obstThrL:
            if obst[5] < obstThrL : 
                # switch from obst avoidance to goal tracking if obstacle got unseen
                obstacle_state="obstacle_passed"
                
    if  obstacle_state=="obstacle_passed":
        # goal tracking: turn toward the goal
        await forward(100)
    else:
        # obstacle avoidance: accelerate wheel near obstacle
        motor_left_target = speed0 + obstSpeedGain * (obst[0] // 100)
        motor_right_target = speed0 + obstSpeedGain * (obst[1] // 100)
        await motorset(motor_left_target,motor_right_target)


#Local navigation 2 : Not used
## Parameters for local navigation
threshold_obst = 3500 
threshold_loc = 2500
local_motor_speed = 100
threshold_obst_list = [3200, 3600, 3600, 3600, 3200]

async def local_navigation():
    threshold_obst = 1000
    threshold_loc = 800
    local_motor_speed = 200
    threshold_obst_list = [3200, 3600, 3600, 3600, 3200]
    sens = await get_proximity_values()

    # Follow the obstacle by the left
    if (sens[0] + sens[1]) > (sens[4] + sens[3]):
        await bypass('right', sens, threshold_loc, local_motor_speed)
        
    # Follow the obstacle by the right    
    else:
        await bypass('left', sens, threshold_loc, local_motor_speed)

async def bypass(leftright, sens, threshold_loc, local_motor_speed):
    global local_obstacle
    if leftright == "right":
        while sum(sens[i] > threshold_obst for i in range(0, 5)) > 0:
            print("Turn right")
            #await rotate(np.pi/6/2, local_motor_speed)
            await motorset(100,-100)
            #await asyncio.sleep(0.2)
            sens = await get_proximity_values()
            print(sens)

        await forward(local_motor_speed)
        time.sleep(2)


        while sens[0] < threshold_loc:
            await motorset(-50,50)
            #await asyncio.sleep(0.2)
            #time.sleep(2)
            sens = await get_proximity_values()
            local_obstacle=False
            
        
        

    elif leftright == "left":
        while sum(sens[i] > threshold_loc for i in range(0, 5)) > 0:
            print("Turn left")
            await rotate(-np.pi/6, local_motor_speed)
            #await asyncio.sleep(0.2)
            sens = await get_proximity_values()

        while sens[4] > threshold_loc:
            await forward(local_motor_speed)
            #await asyncio.sleep(0.2)
            #time.sleep(0.2)
            sens = await get_proximity_values()

    if(leftright=="right" and sens[0] > threshold_loc):
        await rotate(np.pi/2, 100)
    

    await forward(local_motor_speed)
    #time.sleep(2)
    await stop_motor()

# Run the local_navigation function
#await local_navigation()
#test local navigation



