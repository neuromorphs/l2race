# the actual model of car, run on server
# TODO move to separate repo to hide from participants
import logging
from math import sin, radians, degrees, cos, copysign
from typing import Tuple
from scipy.integrate import RK23, RK45, LSODA, BDF, DOP853
from timeit import default_timer as timer
import random
import numpy as np

from src.car_state import car_state
from src.globals import *
from src.l2race_utils import my_logger
from src.track import track

logger = my_logger(__name__)

# import sys
# sys.path.append('../commonroad-vehicle-models/Python')
# either copy/clone https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/Python to commonroad
# or make symlink to folder named commonroad within l2rac
from commonroad.parameters_vehicle1 import parameters_vehicle1  # Ford Escort
from commonroad.parameters_vehicle2 import parameters_vehicle2  # BMW 320i
from commonroad.parameters_vehicle3 import parameters_vehicle3  # VW Vanagon
from .parameters_drifter import parameters_drifter
from commonroad.init_KS import init_KS
from commonroad.init_ST import init_ST
from commonroad.init_MB import init_MB
from commonroad.vehicleDynamics_KS import vehicleDynamics_KS  # kinematic single track, no slip
from commonroad.vehicleDynamics_ST import vehicleDynamics_ST  # single track bicycle with slip
from commonroad.vehicleDynamics_MB import vehicleDynamics_MB  # fancy multibody model


LOGGING_INTERVAL_CYCLES = 0  # 0 to disable # 1000 # log output only this often
MODEL = vehicleDynamics_ST  # vehicleDynamics_KS vehicleDynamics_ST vehicleDynamics_MB
MAX_TIMESTEP = 0.1  # Max timestep of car model simulation. We limit it to avoid instability
SOLVER = RK45  # DOP853 LSODA BDF RK45 RK23 # faster, no overhead but no checking
PARAMETERS = parameters_vehicle2
RTOL = 1e-2
ATOL = 1e-4

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
ISLIPANGLE = 6


class car_model:
    """
    Car model, hidden from participants, updated on server
    """
    def __init__(self,
                 track: track = None,
                 car_name: str = None,
                 client_ip: Tuple[str, int] = None,
                 allow_off_track: bool = False):

        self.n_eval_total = 0  # Number of simulation steps for this car performed since the program was started/reseted
        self.track = track # Track of which car is driving

        # Randomly chose initial position
        x_start, y_start = self.choose_initial_position()
        # Create car_state object - object keeping all the information user can access
        self.car_state:car_state = car_state(x=x_start, y=y_start, body_angle_deg=self.track.start_angle,
                                   name=car_name, client_ip=client_ip)

        # Variables passed_anti_cheat_rect, round_num serves monitoring the completed rounds by the car
        self.passed_anti_cheat_rect = True  # Prohibiting (significant) cutoff
        self.round_num = 0  # Counts the rounds
        self.s_rounds = ''  # String to keep information about completed rounds

        # change MODEL_TYPE to select vehicle model type (vehicle dynamics - how car_state is calculated from car parameters)
        self.model = MODEL  # 'KS' 'ST' 'MB' # model type KS: kinematic single track, ST: single track (with slip), MB: fancy multibody
        if self.model == vehicleDynamics_KS:
            self.model_init = init_KS
            self.model_func = self.func_KS
        elif self.model == vehicleDynamics_ST:
            self.model_init = init_ST
            self.model_func = self.func_ST
        elif self.model == vehicleDynamics_MB:
            self.model_init = init_MB
            self.model_func = self.func_MB

        # select car with next line - determins static parameters of the car: physical dimensions, strength of engine and breaks, etc.
        self.parameters = PARAMETERS()
        # Set parameters of this particular car
        self.car_state.static_info.width_m = self.parameters.w
        self.car_state.static_info.length_m = self.parameters.l
        # set car accel and braking based on car type (not in parameters from commonroad-vehicle-models)
        self.accel_max = self.zeroTo60mpsTimeToAccelG(4) * G  # 5 second 0-60 mph, very quick car is default
        self.brake_max = .9 * G
        if PARAMETERS == parameters_vehicle1:
            self.accel_max = self.zeroTo60mpsTimeToAccelG(
                10.6) * G  # 1992 ford escort https://www.automobile-catalog.com/car/1992/879800/ford_escort_gt_automatic.html
            self.brake_max = .9 * G
        elif PARAMETERS == parameters_vehicle2:  # BMW 320i
            self.accel_max = self.zeroTo60mpsTimeToAccelG(9.5) * G
            self.brake_max = .95 * G
        elif PARAMETERS == parameters_vehicle3:  # VW vanagon
            self.accel_max = self.zeroTo60mpsTimeToAccelG(17.9) * G
            self.brake_max = .8 * G

        # Set parameters and initial values of the model/solver of the car dynamics equations
        sx0 = self.car_state.position_m.x
        sy0 = self.car_state.position_m.y
        delta0 = 0  #
        vel0 = 0
        Psi0 = radians(self.car_state.body_angle_deg)
        dotPsi0 = 0
        beta0 = 0
        initialState = [sx0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]  # initial state for simulation
        if self.model == vehicleDynamics_MB:
            self.model_state = self.model_init(initialState, self.parameters)  # initial state for MB needs params too
        else:
            self.model_state = self.model_init(initialState)  # initial state
        self.cycle_count = 0
        self.time = 0  # "car's clock" - till what time the the simulation was performed
        self.atol = ATOL
        self.atol = self.atol * np.ones(30)
        self.rtol = RTOL
        self.u = [0, 0]
        self.solver = None
        self.first_step = True

        # Set if a car is allowed to leave track or not
        self.allow_off_track = allow_off_track

    def zeroTo60mpsTimeToAccelG(self, time):
        return (60 * 0.447) / time / G

    def update(self, t_sec)->None:
        """
        Updates the model.
        :param t_sec: new time in seconds
        """
        # logger.debug('updating model with dt={:.1f}ms'.format(dt_sec*1000))

        # Update model state:

        # Set server message TODO: write what it is for
        self.car_state.server_msg = ''

        # Check command coming from client
        command = self.car_state.command

        # If client wants to reset the car, reset the car
        # if command.reset_car: self.reset()  # handled by "restart_car" command to server now

        # Go from driver input to commanded steering and acceleration
        accel, steer_vel_rad_per_sec = self.external_to_model_input(command)
        # u0 = steering angle velocity of front wheels
        # u1 = longitudinal acceleration
        self.u = np.array([float(steer_vel_rad_per_sec), float(accel)], dtype='double')

        calculations_time_start = timer() # start counting time required to update the car model
        # Handle first step of simulation i.e. initialisation
        if self.first_step:
            def u_func():
                return self.u

            def model_func(t, y):
                f = self.model_func(t, y, u_func(), self.parameters)
                return f

            self.solver = SOLVER(fun=model_func, t0=self.time, t_bound=1e99,
                                 y0=self.model_state,
                                 first_step=0.00001, max_step=0.01, atol=ATOL, rtol=RTOL)
            self.first_step = False

        # If requested timestep bigger than maximal timestep, make the update for maximal allowed timestep
        # We limit timestep to avoid instability

        dt_sec=t_sec-self.time
        self.time=t_sec

        # tobi commented out because all cars must have same time
        # if dt_sec > MAX_TIMESTEP:
        #     s = 'bounded real dt_sec={:.1f}ms to {:.2f}ms'.format(dt_sec * 1000, MAX_TIMESTEP * 1000)
        #     # logger.info(s)
        #     self.car_state.server_msg += s
        #     dt_sec = MAX_TIMESTEP

        n_eval_start = self.n_eval_total

        t_simulated_start = self.solver.t  # Time on the car's clock at the beginning of the model update
        too_slow = False  # This flag changes to True if it was not possible to perform real-time update of car model
        while self.solver.t < t_simulated_start + dt_sec and not too_slow:
            self.solver.step()
            too_slow = timer() > calculations_time_start + 0.8 * dt_sec
        t_simulated_end = self.solver.t # Time on the car's clock at the end of the model update
        # Calculate the difference on the "car's clock" during this model update
        t_simulated = t_simulated_end - t_simulated_start

        # tobi changed to be the input time only, let's see effect on model
        # self.time = self.solver.t  # Save the time on the "car's clock"

        # TODO: Write what these try-except lines are for? Does n_eval can change during their execution?
        try:
            self.solver.dense_output()  # interpolate
        except RuntimeError:
            pass

        # Save the results of model update to model_state variable
        self.model_state = self.solver.y

        calculations_time_end = timer()  # stop counting time required to update the model

        # Calculate how much time was needed to perform the requested update of the model
        calculations_time = calculations_time_end - calculations_time_start
        # Compare the time required for calculations (calculations_time)
        # with the advance of time on the car's clock (t_simulated)
        if calculations_time > 0.0001:
            n_eval_stop = self.n_eval_total
            n_eval_diff = n_eval_stop - n_eval_start
            s = '{} took {} evals in {:.1f}ms for timestep {:.1f}ms to advance {:.1f}ms {}'.format(self.model.__name__,
                                                                                                   n_eval_diff,
                                                                                                   calculations_time * 1000,
                                                                                                   dt_sec * 1000,
                                                                                                   t_simulated * 1000,
                                                                                                   ('(too_slow)' if too_slow else ''))
            self.car_state.server_msg += '\n' + s

        # make additional constrains for moving foreward and on reverse gear
        self.constrain_speed(command)

        # Constrain position of a car to map
        self.constrain_to_map()

        # If car is off track the forward speed will be set to zero
        # However it can still move backwards
        self.stop_off_track()

        # Car will experience big friction and slowdown if it is in sand region
        self.sand_deceleration()

        # update driver's observed state from model
        # set l2race driver observed car_state from car model
        self.update_car_state(dt_sec, accel)

        s = self.track.car_completed_round(car_model=self)
        if s is not None:
            self.s_rounds = s
        self.car_state.server_msg += '\n' + self.s_rounds

        if LOGGING_INTERVAL_CYCLES > 0 and self.cycle_count % LOGGING_INTERVAL_CYCLES == 0:
            print('\ncar_model.py: cycle {}, dt={:.2f}ms, soln_time={:.3f}ms: {}'.format(self.cycle_count, dt_sec * 1e3,
                                                                                         calculations_time * 1e3,
                                                                                         str(self.car_state)))

        self.cycle_count += 1

    def choose_initial_position(self):
        # Randomly choose initial position
        positions = ['position_1', 'position_2']
        position = random.choice(positions)
        if position == 'position_1':
            (x_start, y_start) = self.track.start_position_1 * M_PER_PIXEL
        else:
            (x_start, y_start) = self.track.start_position_2 * M_PER_PIXEL
        return x_start, y_start

    def computeSteerVelocityRadPerSec(self, commandedSteering: float):
        # based on https://github.com/f1tenth/f1tenth_gym/blob/master/src/racecar.cpp
        diff = commandedSteering - self.model_state[ISTEERANGLE]
        if abs(diff) > radians(.1):
            # bang/bang control: Sets the steering speed to max value in direction to make difference smaller
            steerVel = copysign(self.parameters.steering.v_max, diff)

            # proportional control: Sets the steering speed to in direction to
            # make difference smaller that is proportional to diff/max_steer
            # steerVel=diff/self.parameters.steering.max
        else:
            steerVel = 0
        return steerVel

    def car_name(self):
        if self.car_state:
            return self.car_state.static_info.name
        else:
            return None

    def restart(self):
        logger.info('restarting car named {}'.format(self.car_name()))
        self.__init__(track=self.track,car_name=self.car_name(),client_ip=self.car_state.static_info.client_ip)

    def external_to_model_input(self, command):
        # Compute commanded longitudinal acceleration from throttle and brake input
        accel = self.acceleration_from_input(command)

        # commanded steering angle (not velocity of steering) from driver to model input
        commanded_steering_rad = command.steering * self.parameters.steering.max
        steer_vel_rad_per_sec = self.computeSteerVelocityRadPerSec(commanded_steering_rad)

        return accel, steer_vel_rad_per_sec

    # Compute commanded longitudinal acceleration from throttle and brake input
    def acceleration_from_input(self, command):

        # Get acceleration from breaks - its direction depends on car velocity
        if self.model_state[ISPEED] > 0:
            accel_break = - command.brake * self.brake_max
        elif self.model_state[ISPEED] < 0:
            accel_break = command.brake * self.brake_max
        else:
            accel_break = 0

        if abs(self.model_state[ISPEED])<0.1:
            accel_break = accel_break*abs(self.model_state[ISPEED])*10

        # Get acceleration from throttle - it direction depends on forward/reverse gear
        if not command.reverse:
            # Forward
            accel_throttle = command.throttle * self.accel_max  # TODO BS params, a_max=11.5m/s^2 is bigger than g
        else:
            # Backward
            if self.model_state[ISPEED] > KS_TO_ST_SPEED_M_PER_SEC:  # moving forward too fast and try to go in reverse, then set throttle to zero
                command.throttle = 0
            accel_throttle = -command.throttle * self.accel_max * REVERSE_TO_FORWARD_GEAR

        # Some the two accelerations
        accel = accel_break + accel_throttle

        return accel

    # We impose additional constrains on the speed of a car
    # These constrains prevent car from accelerating through braking
    # and prevent instability when the velocity gets too negative TODO: find better solution for this last point
    def constrain_speed(self, command):
        if command.reverse:
            # Backward
            if self.model_state[ISPEED] < -KS_TO_ST_SPEED_M_PER_SEC:  # TODO: That is only temporary workaround
                self.model_state[ISPEED] = -KS_TO_ST_SPEED_M_PER_SEC

    # Car will experience big friction and slowdown if it is in sand region
    def sand_deceleration(self):
        surface_type = self.track.get_surface_type(x=self.model_state[IXPOS], y=self.model_state[IYPOS])
        if (not self.allow_off_track) and (surface_type >= 8) and (surface_type <= 12):  # 8 and 12 are boundary lines
            self.model_state[ISPEED] = self.model_state[ISPEED] * SAND_SLOWDOWN

    # If car is off track the forward speed will be set to zero
    # However it can still move backwards
    def stop_off_track(self):
        surface_type = self.track.get_surface_type(x=self.model_state[IXPOS], y=self.model_state[IYPOS])
        if (not self.allow_off_track) and (surface_type == 0):
            self.model_state[ISPEED] = self.model_state[ISPEED] * SAND_SLOWDOWN / 4.0

    # update driver's observed state from model
    # set l2race driver observed car_state from car model
    def update_car_state(self, dtSec, accel):
        self.car_state.time += dtSec
        self.car_state.position_m.x = self.model_state[IXPOS]
        self.car_state.position_m.y = self.model_state[IYPOS]
        self.car_state.speed_m_per_sec = self.model_state[ISPEED]
        self.car_state.steering_angle_deg = degrees(self.model_state[ISTEERANGLE])
        self.car_state.body_angle_deg = degrees(self.model_state[IYAW])
        if self.model == vehicleDynamics_ST:
            self.car_state.yaw_rate_deg_per_sec = degrees(self.model_state[IYAWRATE])
            self.car_state.drift_angle_deg = degrees(self.model_state[ISLIPANGLE])
        elif self.model == vehicleDynamics_MB:
            self.car_state.yaw_rate_deg_per_sec = degrees(self.model_state[IYAWRATE])

        self.car_state.velocity_m_per_sec.x = self.car_state.speed_m_per_sec * cos(
            radians(self.car_state.body_angle_deg))
        self.car_state.velocity_m_per_sec.y = self.car_state.speed_m_per_sec * sin(
            radians(self.car_state.body_angle_deg))
        self.car_state.accel_m_per_sec_2.x = accel
        self.car_state.accel_m_per_sec_2.y = 0  # todo, for now and with KS/ST model

    # Constrain position of the car to map
    def constrain_to_map(self):

        if self.model_state[IXPOS] > (SCREEN_WIDTH_PIXELS - 2) * M_PER_PIXEL:
            self.model_state[IXPOS] = (SCREEN_WIDTH_PIXELS - 2) * M_PER_PIXEL
            if self.model_state[ISPEED] > 0:
                self.model_state[ISPEED] = 0
        elif self.model_state[IXPOS] < 0:
            self.model_state[IXPOS] = 0
            if self.model_state[ISPEED] > 0:
                self.model_state[ISPEED] = 0

        if self.model_state[IYPOS] > (SCREEN_HEIGHT_PIXELS - 2) * M_PER_PIXEL:
            self.model_state[IYPOS] = (SCREEN_HEIGHT_PIXELS - 2) * M_PER_PIXEL
            if self.model_state[ISPEED] > 0:
                self.model_state[ISPEED] = 0
        elif self.model_state[IYPOS] < 0:
            self.model_state[IYPOS] = 0
            if self.model_state[ISPEED] > 0:
                self.model_state[ISPEED] = 0

    def func_KS(self, t, x, u, p):
        f = vehicleDynamics_KS(x, u, p)
        self.n_eval_total += 1
        return f

    def func_ST(self, t, x, u, p):
        f = vehicleDynamics_ST(x, u, p)
        self.n_eval_total += 1
        return f

    def func_MB(self, t, x, u, p):
        f = vehicleDynamics_MB(x, u, p)
        self.n_eval_total += 1
        return f
