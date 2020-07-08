import logging

from pygame.math import Vector2

from src.mylogger import mylogger
logger = mylogger(__name__)

class car_state:
    """
    Complete state of car
    """

    def __init__(self, x, y, angle_deg=0.0, length=4, width=2, max_steering=50, max_acceleration=5.0):
        # intrinsic state
        self.position = Vector2(x, y) # x increases to right, y increases downwards
        self.velocity = Vector2(0.0, 0.0) # vx and vy, with y increasing downwards
        self.speed=0
        self.angle_deg = angle_deg # degrees, increases CCW with zero pointing to right
        self.length = length
        self.width = width

        # limits
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_speed = 20
        self.brake_deceleration = 20
        self.free_deceleration = 2

        # current control input
        self.throttle = 0.0
        self.steering = 0.0
        self.brake=0.0

        #track
        self.track_file=None
        self.next_track_vertex_idx=None
        self.distance_from_track_center=0
        self.track_width_here=self.width*4
        self.lap_fraction=0
        self.angle_to_track_deg=0

