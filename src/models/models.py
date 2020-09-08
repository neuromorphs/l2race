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


import torch
from modeling.RNN0.utilis import Sequence, load_pretrained_rnn
import collections
import numpy as np


class RNN_0_model(client_car_model):
    """
    Most basic RNN model predicting only speed based on throttle, break, dt and speed from previous speed
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)
        # Load pretraind RNN
        # Create RNN instance
        # TODO: Find more flexible way to load various RNN architectures
        h1_size: int = 64
        h2_size: int = 64
        self.dt = 0.0
        inputs_list = ['dt', 'cmd.throttle', 'cmd.brake', 'speed']
        outputs_list = ['speed']
        self.net = Sequence(rnn_name='GRU-64H1-64H1', inputs_list=inputs_list, outputs_list=outputs_list)
        # If a pretrained model exists load the parameters from disc and into RNN instance
        # Also evaluate the performance of this pretrained network
        # by checking its predictions on a randomly generated CartPole experiment
        savepathPre = './modeling/RNN0/save/' + 'GRU-8IN-64H1-64H2-3OUT-4.pt' + '.pt'
        load_pretrained_rnn(self.net, savepathPre)
        self.net = self.net.eval()

    def update_state(self, update_enabled:bool, t: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        # This self.time is inherited from car class
        dt = t-self.time
        self.time=t

        if dt < 0:
            logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
            return

        if update_enabled:


            throttle = modeled_car.car_state.command.throttle
            brake = modeled_car.car_state.command.brake
            speed = modeled_car.car_state.speed_m_per_sec

            rnn_input = torch.from_numpy(np.array((dt, throttle, brake, speed))).float().unsqueeze(0).unsqueeze(0)

            new_speed = self.net(predict_len=1, rnn_input=rnn_input, real_time=True)

            # new_speed = real_car.car_state.speed_m_per_sec
            model2real = new_speed/real_car.car_state.speed_m_per_sec
            new_vel_x = real_car.car_state.velocity_m_per_sec.x * model2real
            new_vel_y = real_car.car_state.velocity_m_per_sec.y * model2real

            modeled_car.car_state.position_m.x += dt * new_vel_x
            modeled_car.car_state.position_m.y += + dt * new_vel_y

            modeled_car.car_state.body_angle_deg = real_car.car_state.body_angle_deg


        else:

            modeled_car.car_state = copy.copy(real_car.car_state)  # make copy that just copies the fields
            modeled_car.car_state.command = copy.copy(real_car.car_state.command)  # and our own command

            throttle = modeled_car.car_state.command.throttle
            brake = modeled_car.car_state.command.brake
            speed = modeled_car.car_state.speed_m_per_sec

            rnn_input = torch.from_numpy(np.array((dt, throttle, brake, speed))).float().unsqueeze(0).unsqueeze(0)

            self.net.initialize_sequence(rnn_input=rnn_input, warm_up_len=1, stack_output=False, all_input=True)


class RNN_0_model_2(client_car_model):
    """
    Most basic RNN model predicting only speed based on throttle, break, dt and speed from previous speed
    """

    def __init__(self, car: car = None) -> None:
        super().__init__(car)
        # Load pretraind RNN
        # Create RNN instance
        # TODO: Find more flexible way to load various RNN architectures
        h1_size: int = 64
        h2_size: int = 64
        self.dt = 0.0
        self.net = Sequence(h1_size, h2_size)
        # If a pretrained model exists load the parameters from disc and into RNN instance
        # Also evaluate the performance of this pretrained network
        # by checking its predictions on a randomly generated CartPole experiment
        savepathPre = './modeling/RNN0/save/' + 'MyNetPre' + '.pt'
        load_pretrained_rnn(self.net, savepathPre)
        self.net = self.net.eval()

    def update_state(self, update_enabled:bool, t: float, car_command: car_command, real_car:car, modeled_car: car) -> None:
        """
        Take input time, car_state, and car_command and update the car_state.

        :param update_enabled: True to update model, False to clone the car_state.
        :param t: new time in seconds.
        :param car_command: car_command.
        :param real_car: real car with state we are modeling.
        :param modeled_car: ghost modeled car whose car_state to update.
        """
        # This self.time is inherited from car class
        dt = t-self.time
        self.time=t

        if dt < 0:
            logger.warning('nonmonotonic timestep of {:.3f}s, setting dt=0'.format(dt))
            return

        if update_enabled:


            throttle = modeled_car.car_state.command.throttle
            brake = modeled_car.car_state.command.brake
            speed = modeled_car.car_state.speed_m_per_sec

            rnn_input = torch.from_numpy(np.array((dt, throttle, brake, speed))).float().unsqueeze(0).unsqueeze(0)

            new_speed = self.net(predict_len=1, rnn_input=rnn_input, real_time=True)

            # new_speed = real_car.car_state.speed_m_per_sec
            model2real = new_speed/real_car.car_state.speed_m_per_sec
            new_vel_x = real_car.car_state.velocity_m_per_sec.x * model2real
            new_vel_y = real_car.car_state.velocity_m_per_sec.y * model2real

            modeled_car.car_state.position_m.x += dt * new_vel_x
            modeled_car.car_state.position_m.y += + dt * new_vel_y

            modeled_car.car_state.body_angle_deg = real_car.car_state.body_angle_deg


        else:

            modeled_car.car_state = copy.copy(real_car.car_state)  # make copy that just copies the fields
            modeled_car.car_state.command = copy.copy(real_car.car_state.command)  # and our own command

            throttle = modeled_car.car_state.command.throttle
            brake = modeled_car.car_state.command.brake
            speed = modeled_car.car_state.speed_m_per_sec

            rnn_input = torch.from_numpy(np.array((dt, throttle, brake, speed))).float().unsqueeze(0).unsqueeze(0)

            self.net.initialize_sequence(rnn_input=rnn_input, warm_up_len=1, stack_output=False, all_input=True)


