import math

class DifferentialDrivePIDController:
    def __init__(self, Kp, Ki, Kd, setpoint_angle):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint_angle = setpoint_angle
        self.prev_angle_error = 0
        self.angle_integral = 0

    def compute(self, current_angle):
        # Angle error
        angle_error = self.setpoint_angle - current_angle

        # Angle proportional term
        P_angle = self.Kp * angle_error
        # Angle integral term
        self.angle_integral += angle_error
        I_angle = self.Ki * self.angle_integral
        # Angle derivative term
        D_angle = self.Kd * (angle_error - self.prev_angle_error)
        self.prev_angle_error = angle_error

        # Calculate left and right wheel speeds
        left_wheel_speed = P_angle + I_angle + D_angle
        right_wheel_speed = -(P_angle + I_angle + D_angle)

        return left_wheel_speed, right_wheel_speed