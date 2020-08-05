# the observed state of car as seen by drivers
from typing import Optional, List

from pygame.math import Vector2

from src.car_command import car_command
from src.l2race_utils import my_logger
logger = my_logger(__name__)
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS, M_PER_PIXEL

VERSION='1.0'

class car_state:
    """
    Complete state of car. Updated by hidden model based on control input.

    """

    def __init__(self, x=M_PER_PIXEL * 490, y=M_PER_PIXEL * 615, body_angle_deg=0.0, length_m=4.5, width_m=2.0, max_steering=40, max_acceleration=100.0): # TODO car dimensions are not part of dynamic state, fix
        # todo constructor should put car at starting line, not middle of screen
        # intrinsic state
        # Screen coordinate system is computer vision standard, 0,0 is UL corner and y increases *downwards*.
        # It causes some confusion about angles.
        # (x, y) = tuple starting coordinates in m

        self.time=0
        self.position_m = Vector2(x, y) # x increases to right, y increases downwards
        self.velocity_m_per_sec = Vector2(0.0, 0.0) # vx and vy, with y increasing downwards, i.e. vy>0 means car is moving down the screen
        self.speed_m_per_sec=0.0 # length of velocity vector
        self.accel_m_per_sec_2 = Vector2(0.0, 0.0) # ax and ay *in frame of car*, *along* car body, i.e. ax>0 means car accelerating forwards. ay>0 means rightward acceleration on car body.
        self.steering_angle_deg = 0.0 # degrees of front wheel steering, increases CW/right with zero straight ahead
        self.body_angle_deg = body_angle_deg # degrees, increases CW (on screen!) with zero pointing to right/east
        self.yaw_rate_deg_per_sec = 0.0 # degrees/sec, increases CW on screen
        self.drift_angle_deg=0.0 # drift angle, (beta) relative to heading direction. Zero for no drift. +-90 for drifting entirely sideways. + is drift to left,- to right. TODO check description correct
        self.length_m = length_m # length in meters
        self.width_m = width_m # width in meters

        # current commanded control input
        self.command = car_command()

        self.time_results = []

        # other car(s) # todo add structure to send other car states to client for rendering

        self.other_car_states: List[car_state] = list()  # list of other car_state for other cars

        self.server_msg='' # message from server to be displayed to driver

    def __str__(self):
        s='{}\npos=({:4.1f},{:4.1f})m vel=({:5.1f},{:5.1f})m/s, speed={:6.2f}m/s accel={:6.2f}m/s^2\nsteering_angle={:4.1f}deg body_angle={:4.1f}deg\nyaw_rate={:4.1f}deg/s drift_angle={:4.1f}\nmsg: {}'\
            .format(str(self.command),
                    self.position_m.x,
                    self.position_m.y,
                    self.velocity_m_per_sec.x,
                    self.velocity_m_per_sec.y,
                    self.speed_m_per_sec,
                    self.accel_m_per_sec_2.length(),
                    self.steering_angle_deg,
                    self.body_angle_deg,
                    self.yaw_rate_deg_per_sec,
                    self.drift_angle_deg,
                    self.server_msg)
        return s

    def get_record_headers(self, car):
        import datetime, time, getpass
        header ='# recorded output from l2race\n# format version: {}\n'.format(VERSION)
        header+=datetime.datetime.now().strftime('# Creation_time="%I:%M%p %B %d %Y"\n')  # Tue Jan 26 13:57:06 CET 2016
        header+='# Creation_time="{}" (epoch ms)\n'.format(int(time.time() * 1000.))
        header+='# Username="{}"\n'.format(getpass.getuser())
        header+='# Car_name="{}"\n'.format(car.name)
        header+='# Track_name="{}"\n'.format(car.track.name)
        h=[
            'time',
            'cmd.auto',
            'cmd.steering',
            'cmd.throttle',
            'cmd.brake',
            'cmd.reverse',
            'pos.x',
            'pos.y',
            'vel.x',
            'vel.y',
            'speed',
            'accel.x',
            'accel.y',
            'steering_angle',
            'body_angle',
            'yaw_rate',
            'drift_angle',
            'length',
            'width',
        ]
        for i in h[:-1]:
            header=header+i+','
        header+=h[-1]
        return header

    def get_record_csvrow(self):
        l=[
            self.time,
            1 if self.command.auto else 0,
            self.command.steering,
            self.command.throttle,
            self.command.brake,
            1 if self.command.reverse else 0,
            self.position_m.x,
            self.position_m.y,
            self.velocity_m_per_sec.x,
            self.velocity_m_per_sec.y,
            self.speed_m_per_sec,
            self.accel_m_per_sec_2.x,
            self.accel_m_per_sec_2.y,
            self.steering_angle_deg,
            self.body_angle_deg,
            self.yaw_rate_deg_per_sec,
            self.drift_angle_deg,
            self.length_m,
            self.width_m,
        ]
        s=''
        for v in l[:-1]:
            s=s+('{},'.format(v))
        s=s+('{:f}'.format(l[-1]))
        return s


