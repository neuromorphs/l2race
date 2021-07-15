# driver controller
import logging

from car import car
from car_command import car_command
from controllers.car_controller import car_controller

logger = logging.getLogger(__name__)
MAX_SPEED = 5.0


class pid_next_waypoint_car_controller(car_controller):
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

        # internet values kp=0.1 ki=0.001 kd=2.8
        self.steer_Kp = 0.8
        self.steer_Ki = 0.000015
        self.steer_Kd = 3.8

        self.steer_p_error = 0
        self.steer_i_error = 0
        self.steer_d_error = 0

        self.max_speed = MAX_SPEED

    def read(self, cmd:car_command):
        """
        Computes the next steering angle tying to follow the waypoint list

        :return: car_command that will be applied to the car
        """
        self.car_command = cmd
        waypoint_distance = self.car.track.get_distance_to_nearest_segment(car_state=self.car.car_state,
                                                                           x_car=self.car.car_state.position_m.x,
                                                                           y_car=self.car.car_state.position_m.y)
        self.__update_steering_error(cte=waypoint_distance)
        self.car_command.throttle = 0.1 if self.car.car_state.speed_m_per_sec < self.max_speed else 0
        self.car_command.steering = self.output_steering_angle()
        return self.car_command

    def __update_steering_error(self, cte):
        """
        Calculate the next steering error for P, I and D

        :param cte: Distance from the car to the nearest waypoint segment
        :return: None
        """
        pre_cte = self.steer_p_error
        self.steer_p_error = cte
        self.steer_i_error += cte
        self.steer_d_error = cte - pre_cte

    def output_steering_angle(self):
        """
        Calculate the steering angle using P,I,D constants

        :return: Double. The next steering angle
        """
        angle = - self.steer_Kp * self.steer_p_error - self.steer_Ki * self.steer_i_error - \
               self.steer_Kd * self.steer_d_error
        return angle
