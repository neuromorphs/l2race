# the actual model of car, run on server
# TODO move to separate repo to hide from participants
import logging
from math import sin, radians, degrees, cos, copysign
from scipy.integrate import RK23, RK45

from src.car_command import car_command
from src.car_state import car_state
from src.globals import *
from src.my_logger import my_logger
from src.track import track

logger = my_logger(__name__)
logger.setLevel(logging.DEBUG)

# import sys
# sys.path.append('../commonroad-vehicle-models/Python')
# either copy/clone https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/Python to commonroad
# or make symlink to folder named commonroad within l2rac
from commonroad.parameters_vehicle1 import parameters_vehicle1 # Ford Escort
from commonroad.parameters_vehicle2 import parameters_vehicle2 # BMW 320i
from commonroad.parameters_vehicle3 import parameters_vehicle3 # VW Vanagon
from .parameters_drifter import parameters_drifter
from commonroad.init_KS import init_KS
from commonroad.init_ST import init_ST
from commonroad.init_MB import init_MB
from commonroad.vehicleDynamics_KS import vehicleDynamics_KS # kinematic single track, no slip
from commonroad.vehicleDynamics_ST import vehicleDynamics_ST # single track bicycle with slip
from commonroad.vehicleDynamics_MB import vehicleDynamics_MB # fancy multibody model
from timeit import default_timer as timer

LOGGING_INTERVAL_CYCLES=1000 # log output only this often
MODEL_TYPE='ST' # 'KS', 'ST'
SOLVER=RK23 # faster, no overhead but no checking
PARAMETERS=parameters_vehicle2
RTOL=1e-2
ATOL=1e-4

# indexes into model state
# states
# x0 = x-position in a global coordinate system
# x1 = y-position in a global coordinate system
# x2 = steering angle of front wheels
# x3 = velocity in x-direction
# x4 = yaw angle
# ST model adds two more
# x5 = yaw rate
# x6 = slip angle at vehicle center
IXPOS = 0
IYPOS = 1
ISTEERANGLE = 2
ISPEED = 3
IYAW = 4
IYAWRATE = 5
ISLIPANGLE=6


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
    def __init__(self, track:track=None):
        # self.model=vehicleDynamics_ST()
        self.car_state=car_state()
        self.passed_anti_cheat_rect = True  # Prohibiting (significant) cutoff
        self.round_num = 0 # Counts the rounds
        self.track=track
        # change MODEL_TYPE to select vehicle model type
        self.model_type = MODEL_TYPE # 'KS' 'ST' 'MB' # model type KS: kinematic single track, ST: single track (with slip), MB: fancy multibody
        if self.model_type== 'KS':
            self.model=vehicleDynamics_KS
            self.model_init=init_KS
            self.model_func=func_KS
        elif self.model_type== 'ST':
            self.model=vehicleDynamics_ST
            self.model_init=init_ST
            self.model_func=func_ST
        elif self.model_type== 'MB':
            self.model=vehicleDynamics_MB
            self.model_init=init_MB
            self.model_func=func_MB

        # select car with next line
        self.parameters=PARAMETERS()

        self.car_state.width_m=self.parameters.w
        self.car_state.length_m=self.parameters.l
        # set car accel and braking based on car type (not in parameters from commonroad-vehicle-models)
        self.accel_max=self.zeroTo60mpsTimeToAccelG(4) * G # 5 second 0-60 mph, very quick car is default
        self.brake_max=.9*G
        if PARAMETERS==parameters_vehicle1:
            self.accel_max= self.zeroTo60mpsTimeToAccelG(10.6) * G #1992 ford escort https://www.automobile-catalog.com/car/1992/879800/ford_escort_gt_automatic.html
            self.brake_max=.9*G
        elif PARAMETERS==parameters_vehicle2: # BMW 320i
            self.accel_max=self.zeroTo60mpsTimeToAccelG(9.5)*G
            self.brake_max=.95*G
        elif PARAMETERS==parameters_vehicle3: # VW vanagon
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
        if self.model_type== 'MB':
            self.model_state = self.model_init(initialState,self.parameters)  # initial state for MB needs params too
        else:
            self.model_state = self.model_init(initialState)  # initial state
        self.cycle_count=0
        self.time=0
        self.atol=ATOL
        self.rtol=RTOL
        self.u=[0,0]

        self.solver = None
        self.first_step=True

    def zeroTo60mpsTimeToAccelG(self,time):
        return (60*0.447)/time/G

    def update(self, dtSec, command:car_command):
        if command.reset:
            self.reset()

        self.car_state.server_msg = ''
        self.car_state.command=command

        # compute commanded longitudinal acceleration from throttle and brake input
        if command.throttle>command.brake: # ignore brake
            accel= (command.throttle) * self.accel_max # commanded acceleration from driver # TODO BS params, a_max=11.5m/s^2 is bigger than g
        elif self.model_state[ISPEED]>0:
            accel= (-command.brake) * self.brake_max
        else:
            accel=0

        if command.reverse:
            accel=-accel/4.

        # go from driver input to commanded steering and acceleration
        commandedSteeringRad = command.steering * self.parameters.steering.max  # commanded steering angle (not velocity of steering) from driver
        steerVelRadPerSec=self.computeSteerVelocityRadPerSec(commandedSteeringRad)

        # u0 = steering angle velocity of front wheels
        # u1 = longitudinal acceleration
        self.u=[float(steerVelRadPerSec),float(accel)]
        start=timer()
        if self.first_step:
            def u_func():
                return self.u

            def model_func(t, y):
                f = self.model_func(t, y, u_func(), self.parameters)
                return f

            self.solver=SOLVER(fun=model_func, t0=self.time, t_bound=1e99,
                        y0=self.model_state,
                        first_step=0.01, max_step=0.01, atol=ATOL, rtol=RTOL)
            self.first_step=False

        if dtSec>0.1:
            s='bounded real dtSec={:.1f}ms to 0.1s'.format(dtSec*1000)
            logger.info(s)
            self.car_state.server_msg+=s
            dtSec=0.1

        self.solver.t=self.time
        self.solver.t_old=self.solver.t
        self.solver.bound=self.time+dtSec
        while self.solver.t<self.time+dtSec:
            self.solver.step()
        self.time+=dtSec
        self.model_state = self.solver.y
        end=timer()

        dtSolveSec=end-start
        if dtSolveSec>dtSec/2:
            s='It took {:.1f}ms to solve timestep for timestep of {:.1f}ms'.format(dtSolveSec*1000, dtSec*1000)
            logger.warning(s)
            self.car_state.server_msg+='\n'+s
        # dfdt=self.model(x=self.model_state, uInit=u, p=self.parameters)
        # print('derivatives of state: '+str(dfdt))
        #
        # # Euler step model
        # self.model_state+=np.asarray(dt * np.array(dfdt))

        # # set speed to zero if it comes out negative from Euler braking
        # self.model_state[ISPEED]=max([0,self.model_state[ISPEED]])

        # update driver's observed state from model
        # set l2race driver observed car_state from car model
        self.car_state.position_m.x=self.model_state[IXPOS]
        self.car_state.position_m.y=self.model_state[IYPOS]
        self.car_state.speed_m_per_sec=self.model_state[ISPEED]
        self.car_state.steering_angle_deg=degrees(self.model_state[ISTEERANGLE])
        self.car_state.body_angle_deg=degrees(self.model_state[IYAW])
        if self.model_type=='ST':
            self.car_state.yaw_rate_deg_per_sec=degrees(self.model_state[IYAWRATE])
            self.car_state.drift_angle_deg=degrees(self.model_state[ISLIPANGLE])
        self.car_state.velocity_m_per_sec.x= self.car_state.speed_m_per_sec * cos(radians(self.car_state.body_angle_deg))
        self.car_state.velocity_m_per_sec.y= self.car_state.speed_m_per_sec * sin(radians(self.car_state.body_angle_deg))
        self.car_state.accel_m_per_sec_2.x=accel
        self.car_state.accel_m_per_sec_2.y=0 # todo, for now and with KS/ST model


        # self.constrain_to_map()
        self.locate() # todo is this where we localize ourselves on map?
        # logger.info(self.car_state)

        if self.cycle_count%LOGGING_INTERVAL_CYCLES==0:
            print('\rcar_model.py: cycle {}, dt={:.2f}ms, soln_time={:.3f}ms: {}'.format(self.cycle_count, dtSec * 1e3,dtSolveSec * 1e3,  str(self.car_state)), end='')

        self.cycle_count+=1

        current_surface = self.track.get_surface_type(car_state=self.car_state)
        if not DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK and current_surface == 0:
            logger.info("went off track, resetting car")
            self.reset()

    def computeSteerVelocityRadPerSec(self, commandedSteering:float):
        # based on https://github.com/f1tenth/f1tenth_gym/blob/master/src/racecar.cpp
        diff=commandedSteering-self.model_state[ISTEERANGLE]
        if abs(diff)>radians(.1):
            # bang/bang control: Sets the steering speed to max value in direction to make difference smaller
            steerVel=copysign(self.parameters.steering.v_max,diff)

            # proportional control: Sets the steering speed to in direction to
            # make difference smaller that is proportional to diff/max_steer
            # steerVel=diff/self.parameters.steering.max
        else:
            steerVel=0
        return steerVel

    def locate(self):
        # todo locate ourselves on self.track
        pass

    def reset(self):
        logger.info('resetting car')
        self.__init__(self.track)
        pass

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

