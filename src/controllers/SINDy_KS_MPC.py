import numpy as np
import do_mpc

from car import car
from controllers.car_controller import car_controller
from car_command import car_command

# UNFINISHED! (USES PLACEHOLDER COST FUNCTION)
# TODO implement cost function

# PARAMETERS
MAX_SPEED = 20.0
MAX_STEERING = 0.5  # can be checked in simulator

# MPC PARAMETERS
N_HORIZON = 10
T_STEP = 0.1


class SINDy_KS_MPC(car_controller):
    """
    Implementation of model predictive controller (MPC)
    Uses model learned through sindy_KS.py script (stored in KS_model.csv)
    """

    def __init__(self, my_car: car = None):
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        self.car = my_car
        self.car_command = car_command()
        self.max_speed = MAX_SPEED
        self.max_steering = MAX_STEERING
        self.n_horizon = N_HORIZON
        self.t_step = T_STEP

        # create model object
        self.model = do_mpc.model.Model('continuous')

        # define state space variables
        x = self.model.set_variable(var_type='_x', var_name='x')
        y = self.model.set_variable(var_type='_x', var_name='y')
        speed = self.model.set_variable(var_type='_x', var_name='speed')
        steering_angle = self.model.set_variable(var_type='_x', var_name='steering_angle')
        body_angle = self.model.set_variable(var_type='_x', var_name='body_angle')

        # define commands
        throttle = self.model.set_variable(var_type='_u', var_name='throttle')
        brake = self.model.set_variable(var_type='_u', var_name='brake')
        steering = self.model.set_variable(var_type='_u', var_name='steering')

        # define time-varying parameters for controller
        # distance_to_nearest_segment = self.model.set_variable(var_type='_tvp', var_name='distance_to_nearest_segment')
        # nearest_waypoint_idx = self.model.set_variable(var_type='_tvp', var_name='nearest_waypoint_idx')

        # set the right-hand-side of ODEs; learned coefficents hard-coded from modeling/SINDy/KS_model.csv
        self.model.set_rhs('x', speed * np.cos(body_angle))
        self.model.set_rhs('y', speed * np.sin(body_angle))
        self.model.set_rhs('speed', 2.757 * throttle - 9.071 * brake)
        self.model.set_rhs('steering_angle', -12.385 * steering_angle + 6.255 * steering)
        self.model.set_rhs('body_angle', 0.384 * speed * np.tan(steering_angle))

        self.model.setup()

        # create MPC object
        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': self.t_step,
            'n_robust': 0,
            'store_full_solution': False,
            'store_lagr_multiplier': False,
            'store_solver_stats': []
        }

        self.mpc.set_param(**setup_mpc)

        # set ipopt options
        self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'MA57',  # MA27, ..., MA97, mumps
                                        'ipopt.warm_start_init_point': 'yes'})  # enable warm start

        # tvp_template = self.mpc.get_tvp_template()
        # self.mpc.set_tvp_fun(self.tvp_fun(x, y))

        # define cost function
        lterm = 1/(speed+0.001)
        mterm = 1/(x+0.001)+1/(y+0.001)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        # set penalty factors for inputs
        self.mpc.set_rterm(
             throttle=1e-2,
             brake=1e-2,
             steering=1e-2
        )

        # Lower bounds on states:
        self.mpc.bounds['lower','_x', 'speed'] = 0
        self.mpc.bounds['lower','_x', 'steering_angle'] = -self.max_steering * np.pi
        # Upper bounds on states
        self.mpc.bounds['upper','_x', 'speed'] = self.max_speed
        self.mpc.bounds['upper','_x', 'steering_angle'] = self.max_steering * np.pi

        # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'throttle'] = 0
        self.mpc.bounds['lower','_u', 'brake'] = 0
        self.mpc.bounds['lower','_u', 'steering'] = -1
        # Upper bounds on inputs:
        self.mpc.bounds['upper','_u', 'throttle'] = 1
        self.mpc.bounds['upper','_u', 'brake'] = 1
        self.mpc.bounds['upper','_u', 'steering'] = 1

        # mpc.scaling['_x', 'phi_1'] = 2
        # mpc.scaling['_x', 'phi_2'] = 2
        # mpc.scaling['_x', 'phi_3'] = 2

        # Suppress IPOPT outputs
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        self.mpc.setup()

        # create simulator object
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step=self.t_step)
        self.simulator.setup()

        self.mpc.reset_history()

    # def tvp_fun(self, x, y):
    #     for k in range(self.n_horizon+1):
    #             tvp_template['_tvp',k,'distance_to_nearest_segment'] = self.car.track.get_distance_to_nearest_segment(x=x, y=y)
    #             tvp_template['_tvp',k,'nearest_waypoint_idx'] = self.car.track.get_nearest_waypoint_idx(x=x, y=y)
    #     return tvp_template

    def read(self):
        """
        Computes and returns MPC output

        :return: car_command that will be applied to the car
        """
        # get current state
        x = self.car.car_state.position_m.x
        y = self.car.car_state.position_m.y
        speed = self.car.car_state.speed_m_per_sec
        steering_angle = np.deg2rad(self.car.car_state.steering_angle_deg)
        body_angle = np.deg2rad(self.car.car_state.body_angle_deg)

        # distance_to_nearest_segment = self.car.track.get_distance_to_nearest_segment(x=x, y=y)
        # nearest_waypoint_idx = self.car.track.get_nearest_waypoint_idx(x=x, y=y)

        # create initial state array to be passed to MPC
        s0 = np.array([x, y, speed, steering_angle, body_angle]).reshape(-1,1)

        # pass inital state to MPC-simulator and MPC
        self.simulator.x0 = s0
        self.mpc.x0 = s0

        self.mpc.set_initial_guess()

        # predict horizon and compute optimal commands
        for i in range(self.n_horizon):
            u0 = self.mpc.make_step(s0)
            s0 = self.simulator.make_step(u0)

        # convert commands to float before passing to simulator
        self.car_command.throttle = float(u0[0])
        self.car_command.brake = float(u0[1])
        self.car_command.steering = float(u0[2])
        self.car_command.autodrive_enabled = True

        return self.car_command
