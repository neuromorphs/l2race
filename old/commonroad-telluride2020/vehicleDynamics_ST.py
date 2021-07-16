import math

from src.l2race_settings import KS_TO_ST_SPEED_M_PER_SEC
from .steeringConstraints import steeringConstraints
from .accelerationConstraints import accelerationConstraints
from .vehicleDynamics_KS import vehicleDynamics_KS


def friction_steering_constraint(acceleration, yaw_rate, steering_velocity, velocity, steering_angle, p):
    ''' Moritz Klischat: limits the steering angle based on the current velocity and/or acceleration input.
    Then it should at least not be possible to turn sharply at high speed.

     :param acceleration - longtitudinal acceleration m/s^2
     :param yaw_rate - rad/sec
     :param velocity - speed along body axis m/s
     :param steering_angle - angle of front steering wheels in rad
     :param steering_velocity: The velocity of steering rate change in rad/s
     :param p - the vehicle model parameters

     :returns steering velocity in rad/s
     '''
    if velocity < KS_TO_ST_SPEED_M_PER_SEC:
        return steering_velocity
    # os = steering_velocity
    factor = 1 - abs(2 * velocity / p.longitudinal.v_max)
    if factor < .2: factor = .2
    steering_velocity = steering_velocity * factor
    # print('speed: {:8.2f} old steering: {:8.2f}, new steering: {:8.2f} reduction factor {:8.2f}'.format(velocity,os,steering_velocity,factor))
    return steering_velocity


# @jit(fa(fa, fa, vehicle_params_type))
def vehicleDynamics_ST(x, uInit, p):
    # vehicleDynamics_ST - single-track vehicle dynamics 
    #
    # Syntax:  
    #    f = vehicleDynamics_ST(x,u,p)
    #
    # Inputs:
    #    x - vehicle state vector
    #    u - vehicle input vector
    #    p - vehicle parameter vector
    #
    # Outputs:
    #    f - right-hand side of differential equations
    #
    # Example: 
    #
    # Other m-files required: none
    # Subfunctions: none
    # MAT-files required: none
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      12-January-2017
    # Last update:  16-December-2017
    #               03-September-2019
    # Last revision:---

    # ------------- BEGIN CODE --------------

    # set gravity constant
    g = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = p.tire.p_dy1
    C_Sf = -p.tire.p_ky1 / p.tire.p_dy1
    C_Sr = -p.tire.p_ky1 / p.tire.p_dy1
    lf = p.a
    lr = p.b
    h = p.h_s
    m = p.m
    I = p.I_z

    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity scalar along body (positive forward)
    # x5 = yaw angle
    # x6 = yaw rate
    # x7 = slip angle at vehicle center

    # u1 = steering angle velocity of front wheels
    # u2 = longitudinal acceleration

    # consider steering/acceleration constraints
    u = [
        steeringConstraints(x[2], uInit[0], p.steering),
        accelerationConstraints(x[3], uInit[1], p.longitudinal)
    ]

    u[0] = friction_steering_constraint(u[1], x[5], u[0], x[3], x[4], p)

    # switch to kinematic model for small velocities
    if x[3] < KS_TO_ST_SPEED_M_PER_SEC:  # tobi added for reverse gear and increased to 1m/s to reduce numerical instability at low speed by /speed - hint from matthias
        # wheelbase
        lwb = p.a + p.b

        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        f_ks = vehicleDynamics_KS(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4],
             u[1] / lwb * math.tan(x[2]) + x[3] / (lwb * math.cos(x[2]) ** 2) * u[0],
             0]

    else:
        # system dynamics
        f = [x[3] * math.cos(x[6] + x[4]),
             x[3] * math.sin(x[6] + x[4]),
             u[0],
             u[1],
             x[5],
             -mu * m / (x[3] * I * (lr + lf)) * (
                         lf ** 2 * C_Sf * (g * lr - u[1] * h) + lr ** 2 * C_Sr * (g * lf + u[1] * h)) * x[5] \
             + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h)) * x[6] \
             + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
             (mu / (x[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1) *
             x[5] \
             - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * x[6] \
             + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2]]

    return f

    # ------------- END OF CODE --------------
