# the actual model of car, run on server
# TODO move to separate repo to hide from participants
import logging
from math import sin, radians, degrees, cos, copysign
from typing import Tuple
from scipy.integrate import solve_ivp  # Methods tried before, now not uesed anymore: RK23, RK45, LSODA, BDF, DOP853
from timeit import default_timer as timer
import random
import numpy as np

from src.car_state import car_state
from src.globals import *
from src.l2race_utils import my_logger
from src.track import track
from .car_command import car_command

logger = my_logger(__name__)


# We use the car models from TUM  https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/Python
# get the latest version of the common road models by pulling the sub module: "git submodule update --init --recursive" (recommended)
# The submodule is in the folder commonroad-vehicle-models.
# If you need to make changes, you can also fork the gitlab repository and add your own version as submodule
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1  # Ford Escort
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2  # BMW 320i
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3  # VW Vanagon
from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.init_mb import init_mb
from vehiclemodels.init_std import init_std
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks  # kinematic single track, no slip
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st  # single track bicycle with slip
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb  # fancy multibody model
# drifter
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std  # new drifter model from Gerald Wunsching


LOGGING_INTERVAL_CYCLES = 0  # 0 to disable # 1000 # log output only this often

# indexes into model state
# states, 1 based from commonroads from matlab origins
# x1 = x-position in a global coordinate system
# x2 = y-position in a global coordinate system
# x3 = steering angle of front wheels
# x4 = velocity at vehicle center
# x5 = yaw angle
# x6 = yaw rate
# x7 = slip angle at vehicle center
# x8 = front wheel angular speed
# x9 = rear wheel angular speed

# u1 = steering angle velocity of front wheels
# u2 = longitudinal acceleration

IXPOS = 0
IYPOS = 1
ISTEERANGLE = 2
ISPEED = 3
IYAW = 4
IYAWRATE = 5
ISLIPANGLE = 6
IFWSPEED=7
IRWSPEED=8

class car_model:
    """
    Car model, hidden from participants, updated on server
    """
    def __init__(self,
                 track: track = None,
                 car_name: str = None,
                 client_ip: Tuple[str, int] = None,
                 allow_off_track: bool = False):
        """ Constructs new car model

        :param track: the track to run on
        :param car_name: the name of this car
        :param client_ip: our IP address
        :param allow_off_track: whether to allow car to go offtrack

        """

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
        self.model = MODEL  # 'KS' 'ST' 'MB' 'STD' # model type KS: kinematic single track, ST: single track (with slip), MB: fancy multibody, STD: drifter model added 2020
        if self.model == vehicle_dynamics_ks:
            self.model_init = init_ks
            self.model_func = self.func_ks
        elif self.model == vehicle_dynamics_st:
            self.model_init = init_st
            self.model_func = self.func_st
        elif self.model == vehicle_dynamics_mb:
            self.model_init = init_mb
            self.model_func = self.func_mb
        elif self.model == vehicle_dynamics_std:
            self.model_init = init_std
            self.model_func = self.func_std

        # select car with next line - determins static parameters of the car: physical dimensions, strength of engine and breaks, etc.
        self.parameters = PARAMETERS
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
        elif PARAMETERS == parameters_vehicle4:  # trailer, not supported
            raise RuntimeError('parameters_vehicle4 is a trailer and is not supported currently')

        # Set parameters and initial values of the model/solver of the car dynamics equations
        sx0 = self.car_state.position_m.x
        sy0 = self.car_state.position_m.y
        delta0 = 0  #
        vel0 = 0
        Psi0 = radians(self.car_state.body_angle_deg)
        dotPsi0 = 0
        beta0 = 0
        initialState = [sx0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]  # initial state for simulation
        if self.model == vehicle_dynamics_mb or self.model == vehicle_dynamics_std:
            self.model_state = np.array(self.model_init(initialState, self.parameters))  # initial state for MB and STD needs params too
        else:
            self.model_state = self.model_init(initialState)  # initial state
        self.cycle_count = 0
        self.time = 0  # "car's clock" - till what time the the simulation was performed
        self.atol = ATOL
        self.atol = self.atol * np.ones(30)
        self.rtol = RTOL
        self.u = [0, 0]
        self.solver = SOLVER
        self.first_step = True

        # Set if a car is allowed to leave track or not
        self.allow_off_track = allow_off_track

    def zeroTo60mpsTimeToAccelG(self, time):
        return (60 * 0.447) / time / G

    def update(self, dt_sec)->None:
        """
        Updates the model.
        :param dt_sec: time advance in seconds
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

        calculations_time_start = timer()  # start counting time required to update the car model

        # Prepare functions gathering the car's dynamics to pass them into ode solver
        def u_func():
            return self.u

        def model_func(t, y):
            f = self.model_func(t, y, u_func(), self.parameters)
            return f

        # Save number of calls of the above function
        n_eval_start = self.n_eval_total

        # Integrate equations
        if self.solver=='euler':
            t=self.time
            while t<self.time+dt_sec:
                dt=EULER_TIMESTEP_S if self.time+dt_sec>EULER_TIMESTEP_S else self.time+dt_sec-t
                grad=model_func(t,self.model_state)
                self.model_state+=dt*np.array(grad)
                t+=dt
            t_simulated_end=t

        else:
            self.solver = solve_ivp(fun=model_func,
                                    t_span=[self.time, self.time + dt_sec],
                                    method=SOLVER,
                                    y0=self.model_state,
                                    atol=ATOL,
                                    rtol=RTOL)
            t_simulated_end = self.solver.t[-1]  # True time on the car's clock at the end of the model update
            # Save the results of model update to model_state variable
            self.model_state = self.solver.y[:, -1]

        # check if soln went haywire, i..e. any element of state is NaN
        has_nan=np.isnan(np.sum(self.model_state))
        if has_nan:
            logger.warning(f'model state has NaN: {self.model_state}')
            quit(1)

        # This flag changes to True if it was not possible to perform real-time update of car model
        too_slow = timer() > calculations_time_start + 0.8 * dt_sec



        # Calculate the difference on the "car's clock" during this model update
        t_simulated = t_simulated_end - self.time


        calculations_time_end = timer()  # stop counting time required to update the model


        # Calculate how much time was needed to perform the requested update of the model
        calculations_time = calculations_time_end - calculations_time_start
        # Compare the time required for calculations (calculations_time)
        # with the advance of time on the car's clock (t_simulated)
        if calculations_time > 0.0001:
            n_eval_stop = self.n_eval_total
            n_eval_diff = n_eval_stop - n_eval_start
            too_slow=('(too_slow)' if too_slow else '')
            s=f'{self.model.__name__} with solver {self.solver} took {n_eval_diff} evals in {(calculations_time * 1000):.1f}ms for timestep  {dt_sec * 1000:.1f}ms to advance  {(t_simulated * 1000):.1f}ms  {too_slow }'
            self.car_state.server_msg += '\n' + s


        # make additional constraints for moving foreward and on reverse gear
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

    def external_to_model_input(self, command) -> Tuple[float,float]:
        """
        Computes commanded longitudinal acceleration and steering velocity from throttle and steering inputs.

        :param command: the car_command
        :return: the (acceleration, steering velocity) tuple
        """
        accel = self.acceleration_from_input(command)

        # commanded steering angle (not velocity of steering) from driver to model input
        commanded_steering_rad = command.steering * self.parameters.steering.max
        steer_vel_rad_per_sec = self.computeSteerVelocityRadPerSec(commanded_steering_rad)

        return accel, steer_vel_rad_per_sec

    def acceleration_from_input(self, command:car_command)->float:
        """
        Computes commanded longitudinal acceleration from throttle and brake input.

        :param command: the car_command input

        :returns float acceleration in m/s^2
        """

        # Get acceleration from brakes - its direction depends on car velocity
        if self.model_state[ISPEED] > 0:
            accel_brake = - command.brake * self.brake_max
        elif self.model_state[ISPEED] < 0:
            accel_brake = command.brake * self.brake_max
        else:
            accel_brake = 0

        if abs(self.model_state[ISPEED])<0.1:
            accel_brake = accel_brake*abs(self.model_state[ISPEED])*10

        if getattr(command,'reverse') is None:
            pass

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
        accel = accel_brake + accel_throttle

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
        if self.model != vehicle_dynamics_ks:
            self.car_state.yaw_rate_deg_per_sec = degrees(self.model_state[IYAWRATE])
            self.car_state.drift_angle_deg = degrees(self.model_state[ISLIPANGLE])
            if self.model==vehicle_dynamics_std:
                cycle_per_rad=(1/2*np.pi)
                self.car_state.fw_ang_speed_hz=cycle_per_rad*self.model_state[IFWSPEED] # TODO fix units to hz
                self.car_state.rw_ang_speed_hz=cycle_per_rad*self.model_state[IRWSPEED] # TODO fix units to hz
            elif self.model==vehicle_dynamics_mb:
        #x24 = left front wheel angular speed
        #x25 = right front wheel angular speed
        #x26 = left rear wheel angular speed
        #x27 = right rear wheel angular speed
                cycle_per_rad=(1/2*np.pi)
                self.car_state.fw_ang_speed_hz=cycle_per_rad*(self.model_state[23]+self.model_state[24]) # TODO fix units to hz
                self.car_state.rw_ang_speed_hz=(cycle_per_rad*self.model_state[35]+self.model_state[26]) # TODO fix units to hz


        self.car_state.velocity_m_per_sec.x = self.car_state.speed_m_per_sec * cos(
            radians(self.car_state.body_angle_deg))
        self.car_state.velocity_m_per_sec.y = self.car_state.speed_m_per_sec * sin(
            radians(self.car_state.body_angle_deg))
        self.car_state.accel_m_per_sec_2.x = accel
        self.car_state.accel_m_per_sec_2.y = 0  # todo, for now and with KS/ST model, update for drifter and mb
        self.car_state.body_angle_sin=np.sin(radians(self.car_state.body_angle_deg))
        self.car_state.body_angle_cos=np.cos(radians(self.car_state.body_angle_deg))

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

    def func_ks(self, t, x, u, p):
        f = vehicle_dynamics_ks(x, u, p)
        self.n_eval_total += 1
        return f

    def func_st(self, t, x, u, p):
        f = vehicle_dynamics_st(x, u, p)
        self.n_eval_total += 1
        return f

    def func_mb(self, t, x, u, p):
        f = vehicle_dynamics_mb(x, u, p)
        self.n_eval_total += 1
        return f

    def func_std(self, t, x, u, p):
        f = vehicle_dynamics_std(x, u, p)
        self.n_eval_total += 1
        return f
