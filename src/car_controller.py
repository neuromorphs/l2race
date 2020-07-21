# driver controller
import logging
from src.car import Car
from src.car_command import car_command

logger = logging.getLogger(__name__)


class car_controller:
    """
    car controller: gets state and provides control signal
    """
    def __init__(self, car: Car = None):
        """
        :param car: All car info: car_state and track
        For the controller the user needs to know information about the current car state and its position in the track
        Then, the user should implement a controller that generate a new car command to apply to the car
        """
        self.car = car
        self.car_command = car_command()

    def read(self):
        self.car_command.throttle += 0.2
        if self.car_command.throttle >= 1.0:
            self.car_command.throttle = 1.0

        self.car_command.steering += 0.001

        return self.car_command
