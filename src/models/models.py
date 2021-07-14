# models that take car_command and car_state and produce next car_state.
# these models are local to the client and are used for control in e.g. MPC

from abc import ABC # Abstract Base Class python
import copy

from l2race_utils import my_logger
logger = my_logger(__name__)

from car import car
from car_command import car_command
from car_state import car_state



#######
# Part of RNN-team
import pandas as pd
import torch
from modeling.RNN0.utilis import create_rnn_instance, get_device
from track import pixels2meters, meters2pixels
import collections
import numpy as np
from globals import *
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
        dt = time - self.time
        self.time = time
        if update_enabled:
            if dt < 0:
                logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
                return
            modeled_car.car_state.position_m += dt * modeled_car.car_state.velocity_m_per_sec #+(dt*dt)/2*modeled_car.car_state.accel_m_per_sec_2
            modeled_car.car_state.body_angle_deg += dt * modeled_car.car_state.yaw_rate_deg_per_sec
        else:
            modeled_car.car_state = copy.copy(real_car.car_state)  # make copy that just copies the fields
            modeled_car.car_state.command=copy.copy(real_car.car_state.command) # and our own command

        # logger.info('\nreal car state:  {}\nghost car state: {}'.format(real_car.car_state,modeled_car.car_state))

class SINDy_KS_model(client_car_model):
    """
    Kinematic single-track model learned by SINDy
    Hard coded model is read from modeling/SINDy/KS_model.csv after training

    TODO: implement without hard coding learned model
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)

    def model_acceleration(self, throttle, brake):
        return 2.757 * throttle - 9.071 * brake

    def model_steering_rate(self, steering_angle_deg, cmd_steering):
        return np.rad2deg(-12.385 * np.deg2rad(steering_angle_deg) + 6.255 * cmd_steering)

    def model_yaw_rate(self, speed_m_per_sec, steering_angle_deg):
        return np.rad2deg(0.384 * speed_m_per_sec * np.tan(np.deg2rad(steering_angle_deg)))

    def update_state(self, update_enabled:bool, time: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        dt = time - self.time
        self.time = time
        if update_enabled:
            if dt < 0:
                logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
                return

            modeled_car.car_state.position_m.x += modeled_car.car_state.speed_m_per_sec*np.cos(np.deg2rad(modeled_car.car_state.body_angle_deg)) * dt
            modeled_car.car_state.position_m.y += modeled_car.car_state.speed_m_per_sec*np.sin(np.deg2rad(modeled_car.car_state.body_angle_deg)) * dt

            modeled_car.car_state.speed_m_per_sec += self.model_acceleration(car_command.throttle, car_command.brake) * dt
            modeled_car.car_state.steering_angle_deg += self.model_steering_rate(modeled_car.car_state.steering_angle_deg, car_command.steering) * dt
            modeled_car.car_state.body_angle_deg += self.model_yaw_rate(modeled_car.car_state.speed_m_per_sec, modeled_car.car_state.steering_angle_deg) * dt
        else:
            modeled_car.car_state = copy.copy(real_car.car_state)
            modeled_car.car_state.command = copy.copy(real_car.car_state.command)



class RNN_model(client_car_model):
    """
    Most basic RNN model predicting only speed based on throttle, break, dt and speed from previous speed
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)

        # Parameters. Maybe provide later as arguments?
        self.rnn_full_name = 'GRU-8IN-256H1-256H2-5OUT-0'
        self.path_to_rnn = './modeling/RNN0/save/'
        self.closed_loop_list = ['body_angle_deg', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
        self.closed_loop_enabled = False
        self.device = get_device()
        # Load rnn instance and return lists of input, outputs and its name
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=self.rnn_full_name, path_save=self.path_to_rnn, device=self.device)
        self.dt = 0.0
        self.net = self.net.eval()
        self.real_info = pd.DataFrame({
            'time': None,
            'command.autodrive_enabled': None,
            'command.steering': None,
            'command.throttle': None,
            'command.brake': None,
            'command.reverse': None,
            'position_m.x': None,
            'position_m.y': None,
            'velocity_m_per_sec.x': None,
            'velocity_m_per_sec.y': None,
            'speed_m_per_sec': None,
            'accel_m_per_sec_2.x': None,
            'accel_m_per_sec_2.y': None,
            'steering_angle_deg': None,
            'body_angle_deg': None,
            'yaw_rate_deg_per_sec': None,
            'drift_angle_deg': None
        }, index=[0])



        self.normalization_info = NORMALIZATION_INFO

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
        rnn_output = self.net(rnn_input=rnn_input)
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
            logger.warning('non-monotonic timestep of {:.3f}s, setting dt=0'.format(dt))
            return

        self.real_info['time'] = copy.deepcopy(real_car.car_state.time)
        self.real_info['dt'] = dt
        self.real_info['command.autodrive_enabled'] = copy.deepcopy(real_car.car_state.command.autodrive_enabled)
        self.real_info['command.steering'] = copy.deepcopy(real_car.car_state.command.steering)
        self.real_info['command.throttle'] = copy.deepcopy(real_car.car_state.command.throttle)
        self.real_info['command.brake'] = copy.deepcopy(real_car.car_state.command.brake)
        self.real_info['command.reverse'] = copy.deepcopy(real_car.car_state.command.reverse)
        self.real_info['position_m.x'] = copy.deepcopy(real_car.car_state.position_m.x)
        self.real_info['position_m.y'] = copy.deepcopy(real_car.car_state.position_m.y)
        self.real_info['velocity_m_per_sec.x'] = copy.deepcopy(real_car.car_state.velocity_m_per_sec.x)
        self.real_info['velocity_m_per_sec.y'] = copy.deepcopy(real_car.car_state.velocity_m_per_sec.y)
        self.real_info['speed_m_per_sec'] = copy.deepcopy(real_car.car_state.speed_m_per_sec)
        self.real_info['accel_m_per_sec_2.x'] = copy.deepcopy(real_car.car_state.accel_m_per_sec_2.x)
        self.real_info['accel_m_per_sec_2.y'] = copy.deepcopy(real_car.car_state.accel_m_per_sec_2.y)
        self.real_info['steering_angle_deg'] = copy.deepcopy(real_car.car_state.steering_angle_deg)
        self.real_info['body_angle_deg'] = copy.deepcopy(real_car.car_state.body_angle_deg)
        self.real_info['yaw_rate_deg_per_sec'] = copy.deepcopy(real_car.car_state.yaw_rate_deg_per_sec)
        self.real_info['drift_angle_deg'] = copy.deepcopy(real_car.car_state.drift_angle_deg)

    def normalize_input(self, rnn_input):
        for column in rnn_input:
            if self.normalization_info.iloc[0][column] is not None:
                rnn_input.iloc[0][column] /= self.normalization_info.iloc[0][column]
        return rnn_input

    def denormalize_output(self, rnn_output):
        for column in rnn_output:
            if self.normalization_info.iloc[0][column] is not None:
                rnn_output.iloc[0][column] *= self.normalization_info.iloc[0][column]
        return rnn_output


    def update_modeled_car_from_rnn(self, modeled_car, rnn_output):

        if 'time' in rnn_output.columns:
            modeled_car.car_state.time = rnn_output['time']

        if 'command.autodrive_enabled' in rnn_output.columns:
            modeled_car.car_state.command.autodrive_enabled = rnn_output['command.autodrive_enabled']

        if 'command.steering' in rnn_output.columns:
            modeled_car.car_state.command.steering = rnn_output['command.steering']

        if 'command.throttle' in rnn_output.columns:
            modeled_car.car_state.command.throttle = rnn_output['command.throttle']

        if 'command.brake' in rnn_output.columns:
            modeled_car.car_state.command.brake = rnn_output['command.brake']

        if 'command.reverse' in rnn_output.columns:
            modeled_car.car_state.command.brake = rnn_output['command.reverse']

        if 'position_m.x' in rnn_output.columns:
            modeled_car.car_state.position_m.x = rnn_output['position_m.x']

        if 'position_m.y' in rnn_output.columns:
            modeled_car.car_state.position_m.y = rnn_output['position_m.y']

        if 'velocity_m_per_sec.x' in rnn_output.columns:
            modeled_car.car_state.velocity_m_per_sec.x = rnn_output['velocity_m_per_sec.x']

        if 'velocity_m_per_sec.y' in rnn_output.columns:
            modeled_car.car_state.velocity_m_per_sec.y = rnn_output['velocity_m_per_sec.y']

        if 'accel_m_per_sec_2.x' in rnn_output.columns:
            modeled_car.car_state.accel_m_per_sec_2.x = rnn_output['accel_m_per_sec_2.x']

        if 'accel_m_per_sec_2.y' in rnn_output.columns:
            modeled_car.car_state.accel_m_per_sec_2.y = rnn_output['accel_m_per_sec_2.y']

        if 'speed_m_per_sec' in rnn_output.columns:
            modeled_car.car_state.speed_m_per_sec = rnn_output['speed_m_per_sec']

        if 'steering_angle_deg' in rnn_output.columns:
            modeled_car.car_state.steering_angle_deg = rnn_output['steering_angle_deg']

        if 'body_angle_deg' in rnn_output.columns:
            modeled_car.car_state.body_angle_deg = rnn_output['body_angle_deg']

        if 'yaw_rate_deg_per_sec' in rnn_output.columns:
            modeled_car.car_state.yaw_rate_deg_per_sec = rnn_output['yaw_rate_deg_per_sec']

        if 'drift_angle_deg' in rnn_output.columns:
            modeled_car.car_state.drift_angle_deg = rnn_output['drift_angle_deg']

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


