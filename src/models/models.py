# models that take car_command and car_state and produce next car_state.
# these models are local to the client and are used for control in e.g. MPC

from abc import ABC # Abstract Base Class python
import copy

from src.l2race_utils import my_logger
logger = my_logger(__name__)

from src.car import car
from src.car_command import car_command
from src.car_state import car_state

from modeling.SINDy.sindy import load_model


#######
# Part of RNN-team
import pandas as pd
import torch
from modeling.RNN0.utilis import create_rnn_instance, get_device
from src.track import pixels2meters, meters2pixels
import collections
import numpy as np
from src.globals import *
#######

class client_car_model(ABC):
    """
    Abstract base class for client car models
    """
    def __init__(self, car:car=None) -> None:
        self.time=0 # current time of our model
        self.car=car
        pass

    def update_state(self, update_enabled:bool, t:float, car_command:car_command,  real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        pass


class linear_extrapolation_model(client_car_model):
    """
    Simple model where the car just keeps going in same direction with constant velocity.
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)

    def update_state(self, update_enabled:bool, time: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param time: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        dt= time - self.time
        self.time=time
        if update_enabled:
            if dt<0:
                logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
                return
            modeled_car.car_state.position_m+= dt * modeled_car.car_state.velocity_m_per_sec #+(dt*dt)/2*modeled_car.car_state.accel_m_per_sec_2
            modeled_car.car_state.body_angle_deg+= dt * modeled_car.car_state.yaw_rate_deg_per_sec
        else:
            modeled_car.car_state = copy.copy(real_car.car_state)  # make copy that just copies the fields
            modeled_car.car_state.command=copy.copy(real_car.car_state.command) # and our own command

        # logger.info('\nreal car state:  {}\nghost car state: {}'.format(real_car.car_state,modeled_car.car_state))

class SINDy_model(client_car_model):
    """
    SINDy model
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)
        self.model = load_model('modeling/SINDy/SINDy_model.pkl')

    def update_state(self, update_enabled:bool, t: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        dt=t-self.time
        self.time=t
        if update_enabled:
            if dt<0:
                logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
                return

            new_state = self.model.simulate_step(modeled_car.car_state, real_car.car_state.command, t, dt)
            modeled_car.car_state = new_state
        else:
            modeled_car.car_state = copy.copy(real_car.car_state)
            modeled_car.car_state.command=copy.copy(real_car.car_state.command)



class RNN_model(client_car_model):
    """
    Most basic RNN model predicting only speed based on throttle, break, dt and speed from previous speed
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)

        # Parameters. Maybe provide later as arguments?
        self.rnn_full_name = 'GRU-8IN-64H1-64H2-5OUT-0'
        self.path_to_rnn = './modeling/RNN0/save/'
        self.closed_loop_list = ['body_angle', 'pos.x', 'pos.y', 'vel.x', 'vel.y']
        self.closed_loop_enabled = False
        self.device = get_device()
        # Load rnn instance and return lists of input, outputs and its name
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=self.rnn_full_name, path_save=self.path_to_rnn, device=self.device)
        self.dt = 0.0
        self.net = self.net.eval()
        self.real_info = pd.DataFrame({
            'time': None,
            'cmd.auto': None,
            'cmd.steering': None,
            'cmd.throttle': None,
            'cmd.brake': None,
            'cmd.reverse': None,
            'pos.x': None,
            'pos.y': None,
            'vel.x': None,
            'vel.y': None,
            'speed': None,
            'accel.x': None,
            'accel.y': None,
            'steering_angle': None,
            'body_angle': None,
            'yaw_rate': None,
            'drift_angle': None
        }, index=[0])

        normalization_distance = pixels2meters(np.sqrt((SCREEN_HEIGHT_PIXELS**2) + (SCREEN_WIDTH_PIXELS**2)))
        normalization_velocity = 50.0  # Before from Mark 24
        normalization_acceleration = 5.0  # 2.823157895
        normalization_angle = 180.0
        normalization_dt = 1.0e-2

        self.normalization_info = pd.DataFrame({
            'time': None,
            'dt': normalization_dt,
            'cmd.auto': None,
            'cmd.steering': None,
            'cmd.throttle': None,
            'cmd.brake': None,
            'cmd.reverse': None,
            'pos.x': normalization_distance,
            'pos.y': normalization_distance,
            'vel.x': normalization_velocity,
            'vel.y': normalization_velocity,
            'speed': normalization_velocity,
            'accel.x': normalization_acceleration,
            'accel.y': normalization_acceleration,
            'steering_angle': None,
            'body_angle': normalization_angle,
            'yaw_rate': None,
            'drift_angle': None
        }, index=[0])

        self.rnn_output_previous = None # The rnn predicts next time step. So we have to feed in ghost car its output from previous iteration



    def update_state(self, update_enabled:bool, t: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        self.update_real_info(t, real_car)
        # Check if state not None:
        rnn_input = self.real_info[self.inputs_list]
        if update_enabled and self.closed_loop_enabled and (self.rnn_output_previous is not None):
            self.get_closed_loop_input(rnn_input=rnn_input)
        self.normalize_input(rnn_input)
        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = torch.from_numpy(rnn_input).float().unsqueeze(0).unsqueeze(0).to(self.device)
        rnn_output = self.net.initialize_sequence(rnn_input=rnn_input, all_input=True, stack_output=False)
        rnn_output = list(np.squeeze(rnn_output.detach().cpu().numpy()))
        rnn_output = pd.DataFrame(data=[rnn_output], columns=self.outputs_list)
        self.denormalize_output(rnn_output)

        modeled_car.car_state = copy.deepcopy(real_car.car_state)  # make copy that just copies the fields
        modeled_car.car_state.command = copy.deepcopy(real_car.car_state.command)  # and our own command

        if update_enabled and (self.rnn_output_previous is not None):
            self.update_modeled_car_from_rnn(modeled_car=modeled_car, rnn_output=self.rnn_output_previous)
        else:
            pass

        self.rnn_output_previous = rnn_output

    def update_real_info(self, t: float, real_car: car):

        # This self.time is inherited from car class
        dt = t-self.time
        self.time = t

        if dt < 0:
            logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
            return

        self.real_info['time'] = copy.deepcopy(real_car.car_state.time)
        self.real_info['dt'] = dt
        self.real_info['cmd.auto'] = copy.deepcopy(real_car.car_state.command.autodrive_enabled)
        self.real_info['cmd.steering'] = copy.deepcopy(real_car.car_state.command.steering)
        self.real_info['cmd.throttle'] = copy.deepcopy(real_car.car_state.command.throttle)
        self.real_info['cmd.brake'] = copy.deepcopy(real_car.car_state.command.brake)
        self.real_info['cmd.reverse'] = copy.deepcopy(real_car.car_state.command.reverse)
        self.real_info['pos.x'] = copy.deepcopy(real_car.car_state.position_m.x)
        self.real_info['pos.y'] = copy.deepcopy(real_car.car_state.position_m.y)
        self.real_info['vel.x'] = copy.deepcopy(real_car.car_state.velocity_m_per_sec.x)
        self.real_info['vel.y'] = copy.deepcopy(real_car.car_state.velocity_m_per_sec.y)
        self.real_info['speed'] = copy.deepcopy(real_car.car_state.speed_m_per_sec)
        self.real_info['accel.x'] = copy.deepcopy(real_car.car_state.accel_m_per_sec_2.x)
        self.real_info['accel.y'] = copy.deepcopy(real_car.car_state.accel_m_per_sec_2.y)
        self.real_info['steering_angle'] = copy.deepcopy(real_car.car_state.steering_angle_deg)
        self.real_info['body_angle'] = copy.deepcopy(real_car.car_state.body_angle_deg)
        self.real_info['yaw_rate'] = copy.deepcopy(real_car.car_state.yaw_rate_deg_per_sec)
        self.real_info['drift_angle'] = copy.deepcopy(real_car.car_state.drift_angle_deg)

    def normalize_input(self, rnn_input):
        for column in rnn_input:
            if self.normalization_info.iloc[0][column] is not None:
                rnn_input.iloc[0][column] /= self.normalization_info.iloc[0][column]

    def denormalize_output(self, rnn_output):
        for column in rnn_output:
            if self.normalization_info.iloc[0][column] is not None:
                rnn_output.iloc[0][column] *= self.normalization_info.iloc[0][column]


    def update_modeled_car_from_rnn(self, modeled_car, rnn_output):

        if 'time' in rnn_output.columns:
            modeled_car.car_state.time = rnn_output['time']

        if 'cmd.auto' in rnn_output.columns:
            modeled_car.car_state.command.autodrive_enabled = rnn_output['cmd.auto']

        if 'cmd.steering' in rnn_output.columns:
            modeled_car.car_state.command.steering = rnn_output['cmd.steering']

        if 'cmd.throttle' in rnn_output.columns:
            modeled_car.car_state.command.throttle = rnn_output['cmd.throttle']

        if 'cmd.brake' in rnn_output.columns:
            modeled_car.car_state.command.brake = rnn_output['cmd.brake']

        if 'cmd.reverse' in rnn_output.columns:
            modeled_car.car_state.command.brake = rnn_output['cmd.reverse']

        if 'pos.x' in rnn_output.columns:
            modeled_car.car_state.position_m.x = rnn_output['pos.x']

        if 'pos.y' in rnn_output.columns:
            modeled_car.car_state.position_m.y = rnn_output['pos.y']

        if 'vel.x' in rnn_output.columns:
            modeled_car.car_state.velocity_m_per_sec.x = rnn_output['vel.x']

        if 'vel.y' in rnn_output.columns:
            modeled_car.car_state.velocity_m_per_sec.y = rnn_output['vel.y']

        if 'accel.x' in rnn_output.columns:
            modeled_car.car_state.accel_m_per_sec_2.x = rnn_output['accel.x']

        if 'accel.y' in rnn_output.columns:
            modeled_car.car_state.accel_m_per_sec_2.y = rnn_output['accel.y']

        if 'speed' in rnn_output.columns:
            modeled_car.car_state.speed_m_per_sec = rnn_output['speed']

        if 'steering_angle' in rnn_output.columns:
            modeled_car.car_state.steering_angle_deg = rnn_output['steering_angle']

        if 'body_angle' in rnn_output.columns:
            modeled_car.car_state.body_angle_deg = rnn_output['body_angle']

        if 'yaw_rate' in rnn_output.columns:
            modeled_car.car_state.yaw_rate_deg_per_sec = rnn_output['yaw_rate']

        if 'drift_angle' in rnn_output.columns:
            modeled_car.car_state.drift_angle_deg = rnn_output['drift_angle']

    def get_closed_loop_input(self, rnn_input):
        for closed_loop_input in self.closed_loop_list:
            if closed_loop_input not in self.inputs_list:
                raise ValueError('The requested closed loop input {} is not in the inputs list.'
                                 .format(closed_loop_input))
                return
            if closed_loop_input not in self.outputs_list:
                raise ValueError('The requested closed loop input {} is not in the outputs list.'
                                 .format(closed_loop_input))
                return
            rnn_input[closed_loop_input] = self.rnn_output_previous[closed_loop_input]


