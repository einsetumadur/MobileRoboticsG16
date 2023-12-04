from threading import Timer
import numpy as np
import math
import time

TS = 0.1
REAL_THYMIO_SPEED = 37 #mm/s
REAL_THYMIO_ANGULAR_SPEED = 0.72 #rad/s
COMMAND_MOTOR_FOR_CALIBRATION = 100
STD_SPEED = 3 #8.77 #mm^2/s^2
STD_ANGULAR_SPEED = 0.05 #rad^2/s^2

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

# def get_data(node):
#     return ({"sensor":node["prox.horizontal"],
#              "left_speed":node["motor.left.speed"],
#              "right_speed":node["motor.right.speed"]})

# async def get_speed(client, node):
#     await node.wait_for_variables() # wait for Thymio variables values
#     rt = RepeatedTimer(TS, get_data(node)) # it auto-starts, no need of rt.start()

#     try:
#         await client.sleep(TS)
#     finally:
#         rt.stop()
#         #node.send_set_variables(ln.motors(0, 0))

def speed_estimation(left_speed, right_speed):
    speed_measured = (right_speed + left_speed) / 2
    speed = (speed_measured * REAL_THYMIO_SPEED) / COMMAND_MOTOR_FOR_CALIBRATION
    angular_speed_measured = (right_speed - left_speed) / 2
    angular_speed = (angular_speed_measured * REAL_THYMIO_ANGULAR_SPEED) / COMMAND_MOTOR_FOR_CALIBRATION

    return speed, angular_speed

def ex_kalman_filter(speed, angular_speed, bool_camera, position_camera, previous_state_estimation, previous_covariance_estimation,
                     dt, HT=None, HNT=None, RT=None, RNT=None):
    """
    Estimates the current state using the speed sensor data, the camera position estimation and the previous state
    
    param speed [mm/s], angular_speed [rad/s]
    param position_camera_history: last position coordinates given by the camera
    param previous_state_estimation [x, y, theta, v, w]: previous state a posteriori estimation
    param dt: time step
    
    return state_estimation: new a posteriori state estimation
    return P_estimation: new a posteriori state covariance (incertitude)
    """

    # Initialising the constants
    #Assuming that half of the varance is caused by the measurements and half by perturbations to the states
    r_nu_translation = STD_SPEED / 2 # variance on speed measurement
    r_nu_rotation = STD_ANGULAR_SPEED / 2 # variance on angular speed measurement

    qp = 0.04 # variance on position state in mm chosen arbitrarily: √qp = 0.2
    rp = 1 # variance on position measurement in mm
    rp_angle = 0.02 # variance on angle measurement in rad

    # Q = np.array([[?, 0, 0, 0, 0],
    #               [0, ?, 0, 0, 0],
    #               [0, 0, ?, 0, 0],
    #               [0, 0, 0, ?, 0],
    #               [0, 0, 0, 0, ?]]) # process noise covariance matrix MUST CHANGE
    Q = np.identity(5) * qp

    A = np.array([[1, 0, 0, np.cos(previous_state_estimation[2][0]).item() * dt, 0],
                  [0, 1, 0, np.sin(previous_state_estimation[2][0]).item() * dt, 0],
                  [0, 0, 1, 0, dt],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    ## Prediciton Step, through the previous estimation
    predicted_state_estimation = np.dot(A, previous_state_estimation)
    predicted_state_estimation[2][0] = predicted_state_estimation[2][0] % (2 * math.pi) # normalize the angle between 0 and 2pi    

    predicted_state_estimation_jacobian = np.array([[1, 0, previous_state_estimation[4][0].item() * np.cos(previous_state_estimation[2][0]).item() * dt, 0, 0],
                                                    [0, 1, previous_state_estimation[4][0].item() * np.sin(previous_state_estimation[2][0]).item() * dt, 0, 0],
                                                    [0, 0, 1, 0, dt],
                                                    [0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 1]])
    predicted_covariance_estimation = np.dot(predicted_state_estimation_jacobian,
                                             np.dot(previous_covariance_estimation, predicted_state_estimation_jacobian.T))
    if type(Q) != type(None): predicted_covariance_estimation = predicted_covariance_estimation + Q 
    else: predicted_covariance_estimation

    ## Update Step      
    if bool_camera and position_camera is not None:
        # camera position is available
        y = np.array([[position_camera[0:2]], [speed], [angular_speed]])
        H = np.identity(5)
        R = np.array([[rp, 0, 0, 0, 0],
                      [0, rp, 0, 0, 0],
                      [0, 0, rp_angle, 0, 0],
                      [0, 0, 0, r_nu_translation, 0],
                      [0, 0, 0, 0, r_nu_rotation]]) # process noise covariance matrix
    else:
        # no transition, use only the speed
        y = np.array([[speed], [angular_speed]])
        H = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        R = np.array([[r_nu_translation, 0], [0, r_nu_rotation]]) # process noise covariance matrix

    # innovation / measurement residual
    i = y - np.dot(H, predicted_state_estimation)
    # measurement prediction covariance
    S = np.dot(H, np.dot(predicted_covariance_estimation, H.T)) + R
             
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(predicted_covariance_estimation, np.dot(H.T, np.linalg.inv(S)))
    
    # Updated state and covariance estimate
    state_estimation = predicted_state_estimation + np.dot(K, i)
    #P_estimate = previous_covariance_estimate - np.dot(K, np.dot(H, previous_covariance_estimate))
    P_estimation = np.dot((np.identity(5) - np.dot(K, H)), predicted_covariance_estimation)
     
    return state_estimation, P_estimation

async def get_position(state_estimation, P_estimation, start_time, bool_camera, camera_position, node):
    await node.wait_for_variables()
    #await get_speed(client, node)
    left_speed = node["motor.left.speed"]
    right_speed = node["motor.right.speed"]
    speed, angular_speed = speed_estimation(left_speed, right_speed)

    dt = time.time() - start_time 
    
    state_estimation, P_estimation = ex_kalman_filter(speed, angular_speed, bool_camera, camera_position, state_estimation, P_estimation, dt)
    start_time = time.time()

    return state_estimation, P_estimation, speed, angular_speed, start_time