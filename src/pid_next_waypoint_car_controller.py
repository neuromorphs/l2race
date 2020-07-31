# driver controller
import logging
from src.car import car
from src.car_command import car_command

logger = logging.getLogger(__name__)


class pid_next_waypoint_car_controller:
    """
    car controller: gets state and provides control signal.
    This reference implementation is a basic PID controller that aims for the next waypoint.
    """
    def __init__(self, my_car: car = None):
        """
        :type my_car: car object
        :param car: All car info: car_state and track
        For the controller the user needs to know information about the current car state and its position in the track
        Then, the user should implement a controller that generate a new car command to apply to the car
        """
        self.car = my_car
        self.car_command = car_command()

        # PID constants
        self.steer_Kp = 0.08
        self.steer_Ki = 0.000015
        self.steer_Kd = 2.8

        self.steer_p_error = 0
        self.steer_i_error = 0
        self.steer_d_error = 0

    # Computes the control and returns it as a standard keyboard/joystick command
    def read(self):
        # Get the distance from the car position to the nearest waypoint (Middle of the road)
        waypoint_distance = self.car.track.get_distance_to_nearest_segment(car_state=self.car.car_state,
                                                                           x_car=self.car.car_state.position_m.x,
                                                                           y_car=self.car.car_state.position_m.y)
        # Update the steering error using the distance from the car to the middle os the road
        self.__update_steering_error(cte=waypoint_distance)
        # Get the next steering angle and use it in the car command
        self.car_command.steering = self.output_steering_angle()
        # For this example the throttle is fixed
        self.car_command.throttle = 0.1
        return self.car_command

    def __update_steering_error(self, cte):
        pre_cte = self.steer_p_error

        self.steer_p_error = cte
        self.steer_i_error += cte
        self.steer_d_error = cte - pre_cte

    # Compute the steering angle using the error and the PID constants
    def output_steering_angle(self):
        angle = - self.steer_Kp * self.steer_p_error - self.steer_Ki * self.steer_i_error - \
               self.steer_Kd * self.steer_d_error
        return angle
