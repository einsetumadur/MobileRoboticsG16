import math

def PIDController(angle_error):
    Kp = 30
    #Ki = 3
    #angle_integral = 0
    #compute
    # Angle proportional term
    P_angle = Kp * angle_error
    # Angle integral term
    angle_integral += angle_error
    #I_angle = Ki * angle_integral
    # Calculate left and right wheel speeds
    left_wheel_speed = 100 + P_angle 
    right_wheel_speed = 100 -( P_angle )
    return left_wheel_speed, right_wheel_speed