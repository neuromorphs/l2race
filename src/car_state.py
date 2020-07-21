# the observed state of car as seen by drivers
from pygame.math import Vector2

from car_command import car_command
from src.mylogger import mylogger
logger = mylogger(__name__)
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS, M_PER_PIXEL

class CarState:
    """
    Complete state of car. Updated by hidden model based on control input.

    """

    def __init__(self, x=M_PER_PIXEL * 490, y=M_PER_PIXEL * 615, body_angle_deg=0.0, length_m=4.5, width_m=2.0, max_steering=40, max_acceleration=100.0): # TODO car dimensions are not part of dynamic state, fix
        # todo constructor should put car at starting line, not middle of screen
        # intrinsic state
        # Screen coordinate system is computer vision standard, 0,0 is UL corner and y increases *downwards*.
        # It causes some confusion about angles.
        self.position_m = Vector2(x, y) # x increases to right, y increases downwards
        self.velocity_m_per_sec = Vector2(0.0, 0.0) # vx and vy, with y increasing downwards, i.e. vy>0 means car is moving down the screen
        self.speed_m_per_sec=0.0 # length of velocity vector in track coordinates
        self.steering_angle_deg = 0.0 # degrees of front wheel steering, increases CW/right with zero straight ahead
        self.body_angle_deg = body_angle_deg # degrees, increases CW (on screen!) with zero pointing to right/east
        self.yaw_rate_deg_per_sec = 0.0 # degrees/sec, increases CW on screen
        self.drift_angle_deg=0.0 # drift angle, (beta) relative to heading direction. Zero for no drift. +-90 for drifting entirely sideways. + is drift to left,- to right. TODO check description correct
        self.length = length_m # length in meters
        self.width = width_m # width in meters

        # current commanded control input
        self.command=car_command()

        # track
        self.track_file = None
        self.next_track_vertex_idx = None
        self.distance_from_track_center = 0
        self.track_width_here = self.width*4
        self.lap_fraction = 0
        self.angle_to_track_deg = 0

        # other car(s) # todo add structure to send other car states to client for rendering

        self.other_cars: list = None  # list of other car_state for other cars

        self.server_msg=None # message from server to be displayed to driver

    def __str__(self):
        s='{}\npos=({:4.1f},{:4.1f})m vel=({:5.1f},{:5.1f})m/s, speed={:6.2f}m/s\nsteering_angle={:4.1f}deg body_angle={:4.1f}deg\nyaw_rate={:4.1f}deg/s drift_angle={:4.1f}'\
            .format(str(self.command),
                    self.position_m.x,
                    self.position_m.y,
                    self.velocity_m_per_sec.x,
                    self.velocity_m_per_sec.y,
                    self.speed_m_per_sec,
                    self.steering_angle_deg,
                    self.body_angle_deg,
                    self.yaw_rate_deg_per_sec,
                    self.drift_angle_deg)
        return s


