# abstract base class for all car controllers

from abc import ABC # the python Abstract Base Class

from car import car
from car_command import car_command


class car_controller(ABC):

    def __init__(self, my_car: car = None)->None:
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        self.car = my_car

    def read(self, cmd:car_command)->None:
        """
        Control the car via car_command
        :param cmd: the  .steering, .throttle, .brake, .reverse
        """
        pass
