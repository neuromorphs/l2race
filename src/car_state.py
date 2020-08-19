# the observed state of car as seen by drivers
from typing import Optional, List, Tuple

from pygame.math import Vector2

from src.car_command import car_command
from src.l2race_utils import my_logger
logger = my_logger(__name__)
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS, M_PER_PIXEL

VERSION='1.1'
# history
# 1.0, car_name was in header, each row had length_m and width_m
# 1.1 car_name, length_m, width_m moved to car_state.static_into and are now printed to header here

class car_state:
    """
    Complete state of car. Updated by hidden model based on control input.

    """

    class static_info:
        """
        stuff that doesn't change, but is needed for rendering the car - name, client_ip, and dimensions

        """
        def __init__(self, name:str, client_ip:Tuple[str,int], length_m:float, width_m:float):
            self.name:str=name
            self.client_ip:Tuple[str,int] = client_ip
            self.length_m:float = length_m # length in meters
            self.width_m:float = width_m # width in meters


    def __init__(self, name:str='l2racer', client_ip:Tuple[str,int]=None, length_m:float=4., width_m:float=2.,
                 x:float=0, y:float=0., body_angle_deg:float=0):
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
        self.body_angle_deg = 0.0 # degrees, increases CW (on screen!) with zero pointing to right/east
        self.yaw_rate_deg_per_sec = 0.0 # degrees/sec, increases CW on screen
        self.drift_angle_deg=0.0 # drift angle, (beta) relative to heading direction. Zero for no drift. +-90 for drifting entirely sideways. + is drift to left,- to right.

        self.static_info=self.static_info(name=name,length_m=length_m,width_m=width_m,client_ip=client_ip)

        # self.length_m = length_m # length in meters
        # self.width_m = width_m # width in meters

        # current commanded control input
        self.command = car_command()
        self.command.complete_default()

        self.time_results:List[float] = []

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
        """ :returns: the CVS header lines
        """
        import datetime, time, getpass
        header ='# recorded output from l2race\n# format version: {}\n'.format(VERSION)
        header+=datetime.datetime.now().strftime('# Creation_time="%I:%M%p %B %d %Y"\n')  # Tue Jan 26 13:57:06 CET 2016
        header+='# Creation_time="{}" (epoch ms)\n'.format(int(time.time() * 1000.))
        header+='# Username="{}"\n'.format(getpass.getuser())
        header+='# track_name="{}"\n'.format(car.track.name)
        header+='# car_name="{}"\n'.format(car.car_state.static_info.name)
        header+='# length_m="{}"\n'.format(car.car_state.static_info.length_m)
        header+='# width_m="{}"\n'.format(car.car_state.static_info.width_m)
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
        ]
        for i in h[:-1]:
            header=header+i+','
        header+=h[-1]
        return header

    def get_record_csvrow(self):
        """

        :return: row of CSV file
        """
        l=[
            self.time,
            1 if self.command.autodrive_enabled else 0,
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
        ]
        s=''
        for v in l[:-1]:
            s=s+('{},'.format(v))
        s=s+('{:f}'.format(l[-1]))
        return s


