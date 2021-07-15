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
import logging

from src.car import car
from src.car_command import car_command
from src.controllers.car_controller import car_controller

from src.controllers.neural_mpc_controller_util.neural_mpc import *


logger = logging.getLogger(__name__)
MAX_SPEED = 5.0


class neural_mpc_controller(car_controller):
    """
    This reference implementation is a basic PID controller that aims for the next waypoint.
    For the controller the user needs to know information about the current car state and its position in the track
    Then, the user should implement a controller that generate a new car command to apply to the car
    """
    def __init__(self, my_car: car = None):
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        self.car = my_car
        self.car_command = car_command()
        # self.car_controller = CarController(None, predictor="nn", model_name="Dense-128-128-128-128-invariant-10")
        self.car_controller = CarController(None, predictor="nn", model_name="Dense-128-128-128-128-high-speed")

    def read(self):
        """
        Computes the next steering angle tying to follow the waypoint list

        :return: car_command that will be applied to the car
        """
        self.car_command = car_command()

        self.car_controller.set_state(self.car.car_state, self.car.track)
        next_control_sequence = self.car_controller.control_step()
        # self.car_controller.draw_simulated_history(0, [])

        next_control_input = next_control_sequence[0]

        # print("NEXT CONTROL",  next_control_input)
        
        self.car_command.steering = next_control_input[0]
        self.car_command.throttle = min(next_control_input[1],1)

        return self.car_command

  