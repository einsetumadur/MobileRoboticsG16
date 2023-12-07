import math

def PIDController(angle_error):
    Kp = 1
    Ki = 0.1
    #Kd = 0.01
    #prev_angle_error = 0
    angle_integral = 0
    #compute
    # Angle proportional term
    P_angle = Kp * angle_error
    # Angle integral term
    angle_integral += angle_error
    I_angle = Ki * angle_integral
    # Angle derivative term
    #D_angle = Kd * (angle_error - prev_angle_error)
    #prev_angle_error = angle_error

    # Calculate left and right wheel speeds
    left_wheel_speed = 100 + P_angle + I_angle 
    right_wheel_speed = 100 -( P_angle + I_angle )
    return left_wheel_speed, right_wheel_speed