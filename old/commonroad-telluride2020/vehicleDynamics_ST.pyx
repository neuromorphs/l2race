import logging
from cpython cimport array
import math

from src.l2race_settings import KS_TO_ST_SPEED_M_PER_SEC
from src.l2race_utils import my_logger
logger = my_logger(__name__)
from .steeringConstraints import steeringConstraints
from .accelerationConstraints import accelerationConstraints
from .vehicleDynamics_KS import vehicleDynamics_KS

logger = my_logger(__name__)
logger.setLevel(logging.DEBUG)

import cython
if cython.compiled:
    logger.info("check_cython: {} is compiled Cython.".format(__file__))
else:
    logger.warning("check_cython: {} is still just a slowly interpreted script.".format(__file__))


cpdef float friction_steering_constraint(float acceleration, float yaw_rate, float steering_velocity, float velocity, float steering_angle, object p):
    ''' Moritz Klischat: limits the steering angle based on the current velocity and/or acceleration input. 
    Then it should at least not be possible to turn sharply at high speed.

     :param acceleration - longtitudinal acceleration m/s^2
     :param yaw_rate - rad/sec
     :param velocity - speed along body axis m/s
     :param steering_angle - angle of front steering wheels in rad
     :param p - the vehicle model parameters

     :returns steering velocity in rad/s
     '''
    if velocity<KS_TO_ST_SPEED_M_PER_SEC:
        return steering_velocity
    cdef factor = 1 - abs(2 * velocity / p.longitudinal.v_max)
    if factor < .2: factor = .2
    steering_velocity = steering_velocity * factor
    return steering_velocity



# @jit(fa(fa, fa, vehicle_params_type))
cpdef double[:] vehicleDynamics_ST(double[:] x,double[:] uInit,object p):
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

    #------------- BEGIN CODE --------------

    # set gravity constant
    cdef float g = 9.81  #[m/s^2]

    #create equivalent bicycle parameters
    cdef float mu = p.tire.p_dy1
    cdef float C_Sf = -p.tire.p_ky1/p.tire.p_dy1
    cdef float C_Sr = -p.tire.p_ky1/p.tire.p_dy1
    cdef float lf = p.a
    cdef float lr = p.b
    cdef float h = p.h_s
    cdef float m = p.m
    cdef float I = p.I_z

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity scalar along body (positive forward)
    #x5 = yaw angle
    #x6 = yaw rate
    #x7 = slip angle at vehicle center

    #u1 = steering angle velocity of front wheels
    #u2 = longitudinal acceleration
    cdef array.array u=array.array('d',[steeringConstraints(x[2],uInit[0],p.steering),
                                        accelerationConstraints(x[3],uInit[1],p.longitudinal)])

    u[0] = friction_steering_constraint(u[1], x[5], u[0], x[3], x[4], p)

 # init 'left hand side' output

    # cdef array.array f_template=array.array('d', [])
    cdef array.array f_st, f_ks2


# switch to kinematic model for small velocities
    if x[3] < KS_TO_ST_SPEED_M_PER_SEC: # tobi added for reverse gear and increased to 1m/s to reduce numerical instability at low speed by /speed - hint from matthias
        #wheelbase
        lwb = p.a + p.b
        #system dynamics
        x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        f_ks = vehicleDynamics_KS(x_ks,u,p)
        f_ks2=array.array('d', [f_ks[0],  f_ks[1],  f_ks[2],  f_ks[3],  f_ks[4], u[1]/lwb*math.tan(x[2]) + x[3]/(lwb*math.cos(x[2])**2)*u[0],0])
        return f_ks2
    else:
        #system dynamics
        f_st=array.array('d',
            [x[3]*math.cos(x[6] + x[4]),
            x[3]*math.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
            +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
            +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
            -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
            +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])
        return f_st


    #------------- END OF CODE --------------
