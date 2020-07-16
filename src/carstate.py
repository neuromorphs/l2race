import logging

from pygame.math import Vector2

from src.mylogger import mylogger
logger = mylogger(__name__)

class CarState:
    """
    Complete state of car
    """

    def __init__(self, x, y, angle_deg=0.0, length=20, width=20, max_steering=40, max_acceleration=100.0):
        # intrinsic state
        self.position = Vector2(x, y) # x increases to right, y increases downwards
        self.velocity = Vector2(0.0, 0.0) # vx and vy, with y increasing downwards
        self.speed=0
        self.drift_angle_deg=0 # drift angle, (beta) relative to heading direction. Zero for no drift. +-90 for drifting entirely sideways
        self.angle_deg = angle_deg # degrees, increases CCW with zero pointing to right
        self.length = length
        self.width = width

        # limits
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_speed = 400
        self.brake_deceleration = 200
        self.free_deceleration = 30

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

        # other car(s) # todo add structure to send other car states to client for rendering

    def __str__(self):
        s='speed={:.2f} steering={:.2f} angle_deg={:.2f}'.format(self.speed,self.steering,self.angle_deg)
        return(s)
