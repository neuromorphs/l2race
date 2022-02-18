import math
import numpy as np

from car import car
from controllers.car_controller import car_controller
from l2race_settings import M_PER_PIXEL
from car_command import car_command
from l2race_utils import my_logger

logger = my_logger(__name__)

MAX_SPEED = 55.0  # The controller will try to keep to this speed on  curves
D_MIN = 100  # If the distance in to the track edge is smaller than this (in pixels) do full brake
D_MAX = 500  # If the distance is bigger than this do full throttle
WB = 2.9  # [m] wheel base of vehicle - we don't officially know this
LFC = 45.0  # [m] look ahead distance
K = 0.05 # look forward gain # todo what is this?  Is this proportional gain on steering wheel command?


def calc_distance(x1, y1, x2, y2):
    """
    Calculate the distance between 2 point given the coordinates

    :param x1: Point 1 X coordinate
    :param y1: Point 1 Y coordinate
    :param x2: Point 2 X coordinate
    :param y2: Point 2 Y coordinate
    :return: Double. Distance between point 1 and point 2
    """
    dx = x2-x1
    dy = y2-y1
    distance = math.hypot(dx, dy)
    return distance


class pure_pursuit_controller_v2(car_controller):
    """
    This reference implementation is a pure pursuit controller given a waypoint list.
    For the controller the user needs to know information about the current car state, position in the track and the
    waypoint list
    """

    def __init__(self, my_car: car = None):
        """
        Constructs a new instance

        :param car: All car info: car_state and track
        """
        super().__init__(my_car)
        self.car = my_car
        self.max_speed = MAX_SPEED
        self.d_min = D_MIN
        self.d_max = D_MAX
        self.old_nearest_point_index = 0

    def read(self, cmd: car_command) -> None:
        """
        Computes the next steering angle tying to follow the waypoint list

        :return: car_command that will be applied to the car
        """
        super().read(cmd)
        next_waypoint_id = self.car.track.get_nearest_waypoint_idx(car_state=self.car.car_state,
                                                                   x=self.car.car_state.position_m.x,
                                                                   y=self.car.car_state.position_m.y)

        w_ind = next_waypoint_id
        wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        car_x = self.car.car_state.position_m.x
        car_y = self.car.car_state.position_m.y
        yaw_angle = self.car.car_state.body_angle_deg % 360  # degrees, increases CW (on screen!) with zero pointing to right/east
        yaw_angle_rad = (yaw_angle * math.pi) / 180.0
        rear_x = (car_x - (WB / 2.0) * math.cos(yaw_angle_rad))
        rear_y = (car_y - (WB / 2.0) * math.sin(yaw_angle_rad))

        v = self.car.car_state.speed_m_per_sec
        Lf = K * v + LFC  # update look ahead distance

        distance = math.sqrt((wp_x - car_x) ** 2 + (wp_y - car_y) ** 2)
        #        print("waypoint index = ", w_ind, "distance = ", distance, "Lf =", Lf)
        while distance < Lf:
            w_ind += 1  # += 1 for clockwise track; -= 1 for anticlockwise track
            if w_ind < len(self.car.track.waypoints_x):  # < len(self.car.track.waypoints_x):  for clockwise track; > -1 for anticlockwise track
                wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
                wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
            else:
                w_ind = 0  # 0  for clockwise track; len(self.car.track.waypoints_x) - 1 for anticlockwise track
                wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
                wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
            car_x = self.car.car_state.position_m.x
            car_y = self.car.car_state.position_m.y
            distance = math.sqrt((wp_x - car_x) ** 2 + (wp_y - car_y) ** 2)

        alpha = math.atan2(wp_y - rear_y, wp_x - rear_x) - yaw_angle_rad
        steering_angle = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

        cmd.steering = steering_angle

        # Set throttle
        # Calculate distance to the track edge
        car_pos_map = self.car.track.get_position_on_map(car_state=self.car.car_state)

        hit_pos = self.car.track.find_hit_position(angle=self.car.car_state.body_angle_deg, pos=car_pos_map, dl=2.0)
        if hit_pos is not None:
            d = np.linalg.norm(np.array(hit_pos) - np.array(car_pos_map))
            dd = self.d_max-self.d_min
            if d < self.d_min:
                self.max_speed = 0
            elif d > self.d_max:
                self.max_speed = np.inf
            else:
                self.max_speed = MAX_SPEED

            if self.car.car_state.speed_m_per_sec < self.max_speed:
                cmd.throttle = min((d/dd)-(self.d_min/dd), 1.0)
                cmd.brake = 0
            else:
                cmd.brake = min((-d/dd)+(self.d_max/dd), 1.0)
                cmd.throttle = 0
        else:
            cmd.throttle = 0
            cmd.brake = 0
