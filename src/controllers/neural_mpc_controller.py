'''
#######     README      #######

Author: Florian Bolli

This is a MPC controller, that finds trajectories doing the MPPI aproach and predicting trajectories with a neural network
The standalone project can be found here: https://github.com/Florian-Bolli/CommonRoadMPC

Further information can be found in neural_mpc_controler_util/readme.md

For questions do not hesitate to contact me
Email: bollif@ethz.ch

'''

# driver controller
import importlib
import logging
from typing import Optional

import l2race_settings
from l2race_utils import reload_class_if_modified
from src.car import car
from src.car_command import car_command
from src.controllers.car_controller import car_controller

from src.controllers.neural_mpc_controller_util.neural_mpc import *


logger = my_logger(__name__)


class neural_mpc_controller(car_controller):
    """
    This implementation is a neural MPC controller.
    For the controller the user needs to know information about the current car state and its position in the track
    Then, the user should implement a controller that generate a new car command to apply to the car
    """
    def __init__(self, my_car: car = None):
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        super().__init__(my_car)
        self.car = my_car
        self.g=neural_mpc_settings()
        # self.car_controller = CarController(None, predictor="nn", model_name="Dense-128-128-128-128-invariant-10")
        self.car_controller = CarController(None, predictor="nn", model_name="Dense-128-128-128-128-high-speed")
        self.low_speed_controller:Optional[car_controller]=None
        try:
            mod = importlib.import_module(self.g.LOW_SPEED_CONTROLLER_MODULE)
            controller_class=getattr(mod, self.g.LOW_SPEED_CONTROLLER_CLASS)
            self.low_speed_controller = controller_class(my_car) # set it to a class in globals.py
            self.low_speed_controller.car=my_car # must set car
            logger.info(f'using low speed autodrive controller {self.low_speed_controller}')
        except Exception as e:
            logger.error(f'cannot import AUTODRIVE_CLASS named "{self.g.LOW_SPEED_CONTROLLER_CLASS}" from module AUTODRIVE_MODULE named "{self.g.LOW_SPEED_CONTROLLER_MODULE}", got exception {e}')


    def read(self, cmd:car_command) -> None:
        """
        Computes the next steering angle using the current state using car_controller

        :param cmd: car_command that will be applied to the car
        """
        super().read(cmd)
        self.g:neural_mpc_settings=reload_class_if_modified(self.g)

        # check if speed too low, return zero throttle and steering/brake if so
        speed = self.car.car_state.speed_m_per_sec
        if speed < self.g.MIN_SPEED_MPS:
            if self.low_speed_controller is None:
                logger.warning(f'speed {speed} m/s is below MIN_SPEED_MPS of {self.g.MIN_SPEED_MPS} m/s, zeroing throttle and steering')
                cmd.steering=0
                cmd.throttle=0
                cmd.brake=0
                return
            else:
                # logger.info(f'speed {speed} m/s is below MIN_SPEED_MPS of {self.g.MIN_SPEED_MPS} m/s, using {self.low_speed_controller}')
                self.low_speed_controller.read(cmd)
                return


        self.car_controller.set_state(self.car.car_state, self.car.track)
        next_control_sequence = self.car_controller.control_step()
        # self.car_controller.draw_simulated_history(0, [])

        next_control_input = next_control_sequence[0]

        # print("NEXT CONTROL",  next_control_input)
        
        cmd.steering = next_control_input[0]
        cmd.throttle = min(next_control_input[1],1) # throttle can be negative to brake

    def set_car(self,car:car) ->None:
        super(neural_mpc_controller, self).set_car(car)
        if self.low_speed_controller is not None:
            self.low_speed_controller.set_car(car)