from tdmclient import ClientAsync, aw

class thymio:
    def __init__(self):
        self.__start_robot()
        print(3)


    def __start_robot(self):
        self.client = ClientAsync()
        self._node = aw(self.client.wait_for_node()) #_ = protected #__ = private = shouldn't access node outside of the class
        aw(self._node.lock())
        print(2)

    def __motors(self, l_speed=500, r_speed=500):
        return {
            "motor.left.target": [l_speed],
            "motor.right.target": [r_speed],
        }

    def rotate(self, theta, motor_speed): #theta is in radians
        direction_rot=(theta>=0)-(theta<0)
        aw(self._node.set_variables(self.__motors(motor_speed*direction_rot, -motor_speed*direction_rot)))
        # wait time to get theta 1.44 is the factor to correct
        time=(theta)*100/motor_speed*1.44
        aw(self.client.sleep(time))
        # stop the robot
        aw(self._node.set_variables(self.__motors(0, 0)))
        
    def unlock_robot(self):
        self._node.unlock()

    #def __del__(self):
        #Now unlock the robot: in aseba
        #self._node.unlock()
# Example usage:
thymia = thymio()
thymia.rotate(6.28, 100)
thymia.unlock_robot()