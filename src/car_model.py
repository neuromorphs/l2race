import logging
from math import sin, radians, degrees, cos, copysign

from pygame import Vector2

from accelerationConstraints import accelerationConstraints
from car_input import car_input
from car_state import CarState
from globals import *
from src.mylogger import mylogger
# TODO move to separate repo to hide from participants
from steeringConstraints import steeringConstraints
from track import Track
import numpy as np
from scipy.integrate import solve_ivp, odeint

logger = mylogger(__name__)
logger.setLevel(logging.DEBUG)

# import sys
# sys.path.append('../commonroad-vehicle-models/Python')
# or else put commonroad-vehicle-models somewhere, make symlink to the folder within l2race,
# then add this link as a pycharm src folder
# import parameters, these are functions that must be called with (); see examples below
from parameters_vehicle1 import parameters_vehicle1 # Ford Escort
from parameters_vehicle2 import parameters_vehicle2 # BMW 320i
from parameters_vehicle3 import parameters_vehicle3 # VW Vanagon
from init_KS import init_KS
from init_ST import init_ST
from init_MB import init_MB
from vehicleDynamics_KS import vehicleDynamics_KS # kinematic single track, no slip
from vehicleDynamics_ST import vehicleDynamics_ST # single track bicycle with slip
from vehicleDynamics_MB import vehicleDynamics_MB # fancy multibody model

LOGGING_INTERVAL_CYCLES=20 # log output only this often

# indexes into model state
# states
# x0 = x-position in a global coordinate system
# x1 = y-position in a global coordinate system
# x2 = steering angle of front wheels
# x3 = velocity in x-direction
# x4 = yaw angle
IXPOS = 0
IYPOS = 1
ISTEERANGLE = 2
ISPEED = 3
IYAW = 4


def func_KS(t, x , u, p):
    f = vehicleDynamics_KS(x, u, p)
    return f


def func_ST(t, x , u, p):
    f = vehicleDynamics_ST(x, u, p)
    return f


def func_MB(t, x , u, p):
    f = vehicleDynamics_MB(x, u, p)
    return f


class CarModel:
    """
    Car model, hidden from participants, updated on server
    """
    def __init__(self, track:Track=None):
        # self.model=vehicleDynamics_ST()
        self.car_state=CarState()
        self.track=track
        # change MODEL_TYPE to select vehicle model type
        MODEL_TYPE = 'ST' # 'KS' 'ST' 'MB' # model type KS: kinematic single track, ST: single track (with slip), MB: fancy multibody
        if MODEL_TYPE=='KS':
            self.model=vehicleDynamics_KS
            self.model_init=init_KS
            self.model_func=func_KS
        elif MODEL_TYPE=='ST':
            self.model=vehicleDynamics_ST
            self.model_init=init_ST
            self.model_func=func_ST
        elif MODEL_TYPE=='MB':
            self.model=vehicleDynamics_MB
            self.model_init=init_MB
            self.model_func=func_MB

        # select car with next line
        self.parameters=parameters_vehicle3

        self.car_state.width=self.parameters().w
        self.car_state.length=self.parameters().l
        # set car accel and braking based on car type (not in parameters from commonroad-vehicle-models)
        self.accel_max=G/2
        self.brake_max=G/2
        if self.parameters==parameters_vehicle1:
            self.accel_max= self.zeroTo60mpsTimeToAccelG(10.6) * G #1992 ford escort https://www.automobile-catalog.com/car/1992/879800/ford_escort_gt_automatic.html
            self.brake_max=.9*G
        elif self.parameters==parameters_vehicle2: # BMW 320i
            self.accel_max=self.zeroTo60mpsTimeToAccelG(9.5)*G
            self.brake_max=.95*G
        elif self.parameters==parameters_vehicle3: # VW vanagon
            self.accel_max=self.zeroTo60mpsTimeToAccelG(17.9)*G
            self.brake_max=.8*G
        sx0 = self.car_state.position_m.x
        sy0 = self.car_state.position_m.y
        delta0 = 0
        vel0 = 0
        Psi0 = 0
        dotPsi0 = 0
        beta0 = 0
        initialState = [sx0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]  # initial state for simulation
        if MODEL_TYPE=='MB':
            self.model_state = self.model_init(initialState,self.parameters())  # initial state for MB needs params too
        else:
            self.model_state = self.model_init(initialState)  # initial state
        self.cycle_count=0

    def zeroTo60mpsTimeToAccelG(self,time):
        return (60*0.447)/time/G

    def update(self, dt, input:car_input):
        if input.reset:
            self.reset()

        # compute commanded longitudinal acceleration from throttle and brake input
        if input.throttle>input.brake: # ignore brake
            accel=(input.throttle)*self.accel_max # commanded acceleration from driver # TODO BS params, a_max=11.5m/s^2 is bigger than g
        elif self.model_state[ISPEED]>0:
            accel=(-input.brake)*self.brake_max
        else:
            accel=0

        # go from driver input to commanded steering and acceleration
        commandedSteeringRad = input.steering * self.parameters().steering.max  # commanded steering angle (not velocity of steering) from driver
        steerVelRadPerSec=self.computeSteerVelocityRadPerSec(commandedSteeringRad)

        # u0 = steering angle velocity of front wheels
        # u1 = longitudinal acceleration
        u=[steerVelRadPerSec,accel]

        t=[0,dt]
        sol= solve_ivp(fun=self.model_func, t_span=t, t_eval=[dt], y0=self.model_state, args=(u, self.parameters()))
        if not sol.success:
            logger.warning('solver failed: {}'.format(sol.message))
        self.model_state = sol.y.ravel()

        # # compute derivatives of model state from input
        # dfdt=self.model(x=self.model_state, uInit=u, p=self.parameters())
        # print('derivatives of state: '+str(dfdt))
        #
        # # Euler step model
        # self.model_state+=np.asarray(dt * np.array(dfdt))

        # set speed to zero if it comes out negative from Euler braking
        self.model_state[ISPEED]=max([0,self.model_state[ISPEED]])

        # update driver's observed state from model
        # set l2race driver observed car_state from car model
        self.car_state.position_m.x=self.model_state[IXPOS]
        self.car_state.position_m.y=self.model_state[IYPOS]
        self.car_state.speed_m_per_sec=self.model_state[ISPEED]
        self.car_state.steering_angle_deg=degrees(self.model_state[ISTEERANGLE])
        self.car_state.body_angle_deg=degrees(self.model_state[IYAW])
        self.car_state.yaw_rate_deg_per_sec=0.0 # todo get from solver degrees(dfdt[IYAW])
        self.car_state.velocity_m_per_sec.x= self.car_state.speed_m_per_sec * cos(radians(self.car_state.body_angle_deg))
        self.car_state.velocity_m_per_sec.y= self.car_state.speed_m_per_sec * sin(radians(self.car_state.body_angle_deg))

        # self.constrain_to_map()
        self.locate() # todo is this where we localize ourselves on map?
        # logger.info(self.car_state)

        if self.cycle_count%LOGGING_INTERVAL_CYCLES==0:
            print('\rcar_model.py: cycle {}; dt={:.2f}; {}'.format(self.cycle_count, dt,str(self.car_state)),end='')

        self.cycle_count+=1

    def computeSteerVelocityRadPerSec(self, commandedSteering:float):
        # based on https://github.com/f1tenth/f1tenth_gym/blob/master/src/racecar.cpp
        diff=commandedSteering-self.model_state[ISTEERANGLE]
        if abs(diff)>radians(.1):
            # bang/bang control: Sets the steering speed to max value in direction to make difference smaller
            steerVel=copysign(self.parameters().steering.v_max,diff)

            # proportional control: Sets the steering speed to in direction to
            # make difference smaller that is proportional to diff/max_steer
            # steerVel=diff/self.parameters().steering.max
        else:
            steerVel=0
        return steerVel

    def locate(self):
        # todo locate ourselves on self.track
        pass

    def reset(self):
        logger.info('resetting car')
        self.__init__(self.track)
        pass # todo reset to starting line

    def constrain_to_map(self):

        w = SCREEN_WIDTH_PIXELS * M_PER_PIXEL
        h = SCREEN_HEIGHT_PIXELS * M_PER_PIXEL
        if self.car_state.position_m.x > w:
            self.car_state.position_m.x = w
            self.car_state.velocity_m_per_sec.x = 0
        elif self.car_state.position_m.x < 0:
            self.car_state.position_m.x = 0
            self.car_state.velocity_m_per_sec.x = 0
        if self.car_state.position_m.y > h:
            self.car_state.position_m.y = h
            self.car_state.velocity_m_per_sec.y = 0
        elif self.car_state.position_m.y < 0:
            self.car_state.position_m.y = 0
            self.car_state.velocity_m_per_sec.y = 0

        self.car_state.speed_m_per_sec=self.car_state.velocity_m_per_sec.length()

        self.model_state[IXPOS]=self.car_state.position_m.x
        self.model_state[IYPOS]=self.car_state.position_m.y
        self.model_state[ISPEED]=self.car_state.speed_m_per_sec

