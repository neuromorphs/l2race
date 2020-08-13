# driver controller
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
from src.car import car
from src.car_command import car_command
from src.user_input import user_input



class my_controller:
    """
    car controller: gets state and provides control signal.
    This reference implementation is a basic P controller that aims for the next waypoint.

    """
    def __init__(self, my_car: car = None):
        """
        :type my_car: car object
        :param car: All car info: car_state and track
        For the controller the user needs to know information about the current car state and its position in the track
        Then, the user should implement a controller that generate a new car command to apply to the car
        """
        self.car = my_car
        self.car_command:car_command = car_command()
        self.user_input:user_input = user_input()

        self.angle = 0

        self.factor_angle = 0.01
        self.factor_distance = 0.01

    def read(self) -> Tuple[car_command,user_input]:

        self.car_command = car_command()
        '''computes the control and returns it as a standard keyboard/joystick command'''

        segment_distance = self.car.track.get_distance_to_nearest_segment(car_state=self.car.car_state)
        segment_angle = self.car.track.get_current_angle_to_road(car_state=self.car.car_state)
        # print('segment distance: {}'.format(segment_distance))
        # print('segment angle: {}'.format(segment_angle))

        steering_from_distance = -segment_distance*self.factor_distance

        if (segment_angle<0 and segment_distance<0) or (segment_angle>0 and segment_distance>0):
            self.angle = -segment_angle
            steering_from_angle = self.factor_angle * abs(segment_distance) * \
                                        (np.sign(self.angle)*np.sqrt(abs(self.angle)))
        else:
            self.angle = self.factor_angle*segment_angle
            steering_from_angle = self.factor_angle * abs(segment_distance) * \
                                        (np.sign(self.angle)*np.sqrt(abs(self.angle)))

        self.car_command.steering = steering_from_distance+steering_from_angle

        if self.car.car_state.speed_m_per_sec<12.0:
            self.car_command.throttle = 1.0

        return self.car_command, self.user_input

