class thymio_robot:
    def __init__(self):
        pass

    def motors(self,node, l_speed=500, r_speed=500):
        return {
            "motor.left.target": [l_speed],
            "motor.right.target": [r_speed],
        }

    async def rotate(self,node, theta, motor_speed): #theta is in radians
        direction_rot=(theta>=0)-(theta<0)
        await node.set_variables(self.motors(motor_speed*direction_rot, -motor_speed*direction_rot))
        # wait time to get theta 1.44 is the factor to correct
        time=(theta)*100/motor_speed*1.44
        await self.client.sleep(time)
        # stop the robot
        await self.node.set_variables(self.__motors(0, 0))
        # Initialization of Thymio parameters
        # Radius of the wheel
        #R = 20 
        # Distance between wheel axes
        #L = 105 
        
    #def unlock_robot(self):
        #self._node.unlock()

    #def __del__(self):
        #ow unlock the robot: in aseba
        #self._node.unlock()
