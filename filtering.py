from threading import Timer
import numpy as np
import math
import time

TS = 0.1
REAL_THYMIO_SPEED = 37.7 #mm/s
#REAL_THYMIO_ANGULAR_SPEED = 0.75 #rad/s
REAL_THYMIO_ANGULAR_SPEED = 0.70#rad/s
COMMAND_MOTOR_FOR_CALIBRATION = 100
STD_SPEED = 3 #mm^2/s^2
STD_ANGULAR_SPEED = 0.04 #rad^2/s^2
qp =0.04
rp = 0.25

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

def speed_estimation(left_speed, right_speed):
    speed_measured = (right_speed + left_speed) / 2
    speed = (speed_measured * REAL_THYMIO_SPEED) / COMMAND_MOTOR_FOR_CALIBRATION #NOUS
    #speed = (speed_measured * COMMAND_MOTOR_FOR_CALIBRATION) / REAL_THYMIO_SPEED #PROF
    angular_speed_measured = (right_speed - left_speed) / 2
    angular_speed = (angular_speed_measured * REAL_THYMIO_ANGULAR_SPEED) / COMMAND_MOTOR_FOR_CALIBRATION
    #angular_speed = (angular_speed_measured * COMMAND_MOTOR_FOR_CALIBRATION) / REAL_THYMIO_ANGULAR_SPEED
    #print("angular speed measured", angular_speed_measured, "angular speed", angular_speed, "speed measured", speed_measured, "speed", speed)
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
    q_nu_translation = STD_SPEED / 2 # variance on speed measurement
    r_nu_translation = STD_SPEED / 2 # variance on speed measurement
    r_nu_rotation = STD_ANGULAR_SPEED / 2 # variance on angular speed measurement
    q_nu_rotation = STD_ANGULAR_SPEED / 2 # variance on angular speed measurement

    rp = 10 # variance on position measurement in mm
    rp_angle = 0.02 # variance on angle measurement in rad

    Q = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, q_nu_translation, 0],
                  [0, 0, 0, 0, q_nu_rotation]]) # process noise covariance matrix MUST CHANGE
    #Q = np.identity(5) * qp

    theta = previous_state_estimation[2]

    A = np.array([[1, 0, 0, np.cos(theta).item() * dt, 0],
                  [0, 1, 0, np.sin(theta).item() * dt, 0],
                  [0, 0, 1, 0, dt],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    #print("previous_state_estimation angle", previous_state_estimation[2])
    ## Prediciton Step, through the previous estimation
    predicted_state_estimation = np.dot(A, previous_state_estimation)
    #print("predicted_state_estimation angle", predicted_state_estimation[2])

    predicted_state_estimation_jacobian = np.array([[1, 0, - previous_state_estimation[3].item() * np.sin(theta).item() * dt, np.cos(theta).item() * dt, 0], 
                                                    [0, 1, previous_state_estimation[3].item() * np.cos(theta).item() * dt, np.sin(theta).item() * dt, 0],
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
        y = np.array([[position_camera[0]],[position_camera[1]], [position_camera[2]], [speed], [angular_speed]])
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
    #print("i", i)
    # measurement prediction covariance
    S = np.dot(H, np.dot(predicted_covariance_estimation, H.T)) + R
    #print("predicted_covariance_estimation", predicted_covariance_estimation)
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(predicted_covariance_estimation, np.dot(H.T, np.linalg.inv(S)))
    #print("K", K)
    # Updated state and covariance estimate
    state_estimation = predicted_state_estimation + np.dot(K, i)
    state_estimation[2] = state_estimation[2] % (2 * math.pi) 
    
    # normalize the angle between 0 and 2pi    
    #print("np.dot(K, i)", np.dot(K, i))
    #print("new state_estimation angle", state_estimation[2])
    P_estimation = np.dot((np.identity(5) - np.dot(K, H)), predicted_covariance_estimation)
     
    return state_estimation, P_estimation

async def get_position(state_estimation, P_estimation, start_time, bool_camera, camera_position, node, TS):
    print(bool_camera)
    await node.wait_for_variables()
    #await get_speed(client, node)
    left_speed = node["motor.left.speed"]
    right_speed = node["motor.right.speed"]
    speed, angular_speed = speed_estimation(left_speed, right_speed)
    #print("speed", speed, "angular speed", angular_speed, "state_estimation angle", state_estimation[2], "dt", time.time() - start_time)
    dt = time.time() - start_time 
    
    state_estimation, P_estimation = ex_kalman_filter(speed, angular_speed, bool_camera, camera_position, state_estimation, P_estimation, dt)
    start_time = time.time()
    #print("theta", state_estimation[2] * 180 / math.pi, "omega", state_estimation[4])

    return state_estimation, P_estimation, speed, angular_speed, start_time


async def get_position2(state_estimation, P_estimation, TS, bool_camera, camera_position, node, conv1, conv2):
    await node.wait_for_variables()
    #await get_speed(client, node)
    left_speed = node["motor.left.speed"]
    right_speed = node["motor.right.speed"]
    #speed, angular_speed = speed_estimation(left_speed, right_speed)
    #print("speed", speed, "angular speed", angular_speed, "state_estimation angle", state_estimation[2], "dt", time.time() - start_time)
    #dt = time.time() - start_time 
    
    state_estimation, P_estimation,speed,angular_speed = kalman_all(left_speed, right_speed,camera_position, state_estimation, P_estimation, bool_camera,conv1, conv2,TS)
    start_time = time.time()
    #print("theta", state_estimation[2] * 180 / math.pi, "omega", state_estimation[4])

    return state_estimation, P_estimation, speed, angular_speed  #, start_time


def kalman_all(speed_l,speed_r,x_detected,x_est_prev,P_est_prev,detection,speed_conv_factor,speed_thymio_to_rad_s,Ts):
   lin_speed = (speed_l+speed_r)/2
   ang_speed = speed_r-speed_l
   alpha = x_est_prev[3][0]
   A = np.array([[1,0,Ts*np.cos(alpha),0,0],
                [0,1,Ts*np.sin(alpha),0,0],
                [0,0,1,0,0],
                [0,0,0,1,Ts],
                [0,0,0,0,1]])

   Q = np.eye(5)*qp

   x_est_a_priori = np.dot(A, x_est_prev)#+inc_alpha;
   x_est_a_priori[3]=x_est_a_priori[3]%(2*math.pi)

   P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T));
   P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori

   if (detection):
       # transition detected
       pos_last_cam = x_detected;
       y = np.array([[pos_last_cam[0]],[pos_last_cam[1]],[lin_speed*speed_conv_factor],[pos_last_cam[2]],[ang_speed*speed_thymio_to_rad_s]])
       H = np.eye(5)
       #R = np.array([[rp, 0],[0, r_nu]])
       R= np.eye(5)*rp #We don't have precise values yet
   else:
       # no transition, use only the speed
       y = np.array([[lin_speed*speed_conv_factor],[ang_speed*speed_thymio_to_rad_s]])
       H = np.array([[0., 0., 1., 0.,0.],[0., 0., 0., 0.,1.]])
       R = np.eye(2)*rp

   # innovation / measurement residual
   i = y - np.dot(H, x_est_a_priori);

   #print("P_est_a_priori",np.dot(P_est_a_priori, H.T))

   # measurement prediction covariance
   S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R;

   # Kalman gain (tells how much the predictions should be corrected based on the measurements)
   K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)));

   # a posteriori estimate
   x_est = x_est_a_priori + np.dot(K,i);
   P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori));

   return x_est, P_est,lin_speed,ang_speed