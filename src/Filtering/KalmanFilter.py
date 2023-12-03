import numpy as np

def kalman_filter(speed, x_prev, y_prev, x, y, pos_last_trans, x_est_prev, P_est_prev, Ts,
                  HT=None, HNT=None, RT=None, RNT=None):
    """
    Estimates the current state using input sensor data and the previous state
    
    param speed: measured speed (Thymio units)
    param x_prev: previous x position
    param y_prev: previous y position
    param x: current x position
    param y: current y position
    param pos_last_trans: position of the last transition detected by the ground sensor
    param x_est_prev: previous state a posteriori estimation
    param P_est_prev: previous state a posteriori covariance
    
    return pos_last_trans: updated if a transition has been detected
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance (incertitude)
    """

    std_speed = 12     #std_speed = np.std(speed_history)
    q_nu = std_speed / 2 # variance on speed state
    r_nu = std_speed / 2 # variance on speed measurement 

    #The standard deviation is chosen arbitrarily: √qp = 0.2
    qp = 0.04 # variance on position state
    rp = 0.25 # variance on position measurement
    #Assuming that half og the varance is caused by the measurements and half is caused by perturbations to the states
    q_nu = speed/2 # variance on speed state
    r_nu = speed/2 # variance on speed measurement

    # Initialising the remaining constants
    # units: length [mm], time [s]
    A = np.array([[1, Ts], [0, 1]])
    stripe_width = 50
    Q = np.array([[qp, 0], [0, q_nu]])
    speed_conv_factor = 0.3375
    transition_thresh = 500
    
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    x_est_a_priori = np.dot(A, x_est_prev)
    
    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori
    
    ## Update         
    # y, C, and R for a posteriori estimate, depending on transition
    dist_prev = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
    if dist_prev > transition_thresh:
        stripe_width = 44 if y > y_prev else 4  # Update stripe_width based on transition direction
        pos_last_trans += stripe_width
        
        y = np.array([[pos_last_trans], [speed * speed_conv_factor]])
        H = np.array([[1, 0], [0, 1]])
        R = np.array([[rp, 0], [0, r_nu]])
    else:
        # no transition, use only the speed
        y = speed * speed_conv_factor
        H = np.array([[0, 1]])
        R = r_nu

    # innovation / measurement residual
    i = y - np.dot(H, x_est_a_priori)
    # measurement prediction covariance
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
             
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))
    
    
    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K,i)
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))
     
    return pos_last_trans, x_est, P_est