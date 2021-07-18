# abstract base class for all car controllers

from abc import ABC # the python Abstract Base Class

from car import car
from car_command import car_command
from l2race_utils import my_logger

logger=my_logger(__name__)

class car_controller(ABC):

    def __init__(self, my_car: car = None)->None:
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        self.car = my_car

    def read(self, cmd:car_command)->None:
        """
        Control the car via car_command. read sets the values of cmd.steering, cmd.throttle, cmd.brake, and cmd.reverse
        :param cmd: the  .steering, .throttle, .brake, .reverse
        """
        if self.car is None:
            logger.error(f'car is None, {self} cannot control')

    def set_car(self,car:car)->None:
        """Sets the car
        :param car: car object
        """
        self.car=car