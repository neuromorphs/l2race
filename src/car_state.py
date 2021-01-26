# the observed state of car as seen by drivers
from typing import Optional, List, Tuple

from pygame.math import Vector2

from src.car_command import car_command
from src.l2race_utils import my_logger
from inspect import getmembers

logger = my_logger(__name__)
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS, M_PER_PIXEL

VERSION='2.0'
# history
# 1.0, car_name was in header, each row had length_m and width_m
# 1.1 car_name, length_m, width_m moved to car_state.static_into and are now printed to header here
# 2.0 changed to scheme where fields are defined in car_state self.csv_fields and are complete field names

class car_state:
    """
    Complete state of car. Updated by hidden model based on control input.
    Participants have access only to this state, which does not include many of the state variables for some models.

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
        """
        Make a new car_state instance.

        :param name: car name, generated automatically by default in my_args
        :param client_ip: our IP address as (hostname, port)
        :param length_m: car length in meters
        :param width_m: car width in meters
        :param x: starting location x in meters, starting left side
        :param y: starting position y in meters, starting top
        :param body_angle_deg: starting body angle in degrees, 0 is pointing right, position clockwise
        """
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
        self.fw_ang_speed_hz=0.0 # front wheel rotation rate in Hz  - only for drifter std and mb
        self.rw_ang_speed_hz=0.0 # rear wheel rotation rate in Hz  - only for drifter std and mb

        self.static_info=self.static_info(name=name,length_m=length_m,width_m=width_m,client_ip=client_ip)

        # added to state to deal with 2*Pi-cut problem of angle
        self.body_angle_sin = 0.0 # sine of body angle
        self.body_angle_cos = 1.0 # cos of body angle

        # current commanded control input
        self.command = car_command()

        self.time_results:List[float] = []

        self.server_msg='' # message from server to be displayed to driver

        # define all fields to be writtent to CSV file.
        # If field is Vector 2 it will be expanded to x,y parts.
        # If field is bool it will be written as 0,1 for False,True
        self.csv_fields=[
            'time',
            'command.steering',
            'command.throttle',
            'command.brake',
            'command.reverse',
            'command.autodrive_enabled',
            'position_m.x',
            'position_m.y',
            'velocity_m_per_sec.x',
            'velocity_m_per_sec.y',
            'speed_m_per_sec',
            'accel_m_per_sec_2.x',
            'accel_m_per_sec_2.y',
            'steering_angle_deg',
            'body_angle_deg',
            'yaw_rate_deg_per_sec',
            'drift_angle_deg',
            'body_angle_sin',
            'body_angle_cos',
        ]

    """
        Convenience for getting car name from static info

        :return: car name
        """
    def name(self) -> str:
        return self.static_info.name

    def hostname(self):
        """
        Returns hostname of car

        :return: hostname, or None if not yet set
        """
        return self.static_info.client_ip[0] if self.static_info.client_ip else None

    def __str__(self):
        s='t={:1f}\n{}\npos=({:4.1f},{:4.1f})m vel=({:5.1f},{:5.1f})m/s, speed={:6.2f}m/s accel={:6.2f}m/s^2\nsteering_angle={:4.1f}deg body_angle={:4.1f}deg\nyaw_rate={:4.1f}deg/s drift_angle={:4.1f}\nFW {:.1f}Hz RW {:.1f}Hz\nmsg: {}' \
            .format(self.time,
                    str(self.command),
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
                    self.fw_ang_speed_hz,
                    self.rw_ang_speed_hz,
                    self.server_msg)
        return s

    def get_csv_file_header(self, car):
        """
            Outputs the complete CSV file header, including important CSV file header row that has all the fields defineed by self.csv_fields
            :returns: the CVS header lines (# comments plus the CSV header row)
        """
        import datetime, time, getpass
        header ='# recorded output from l2race\n# format version: {}\n'.format(VERSION)
        header+=datetime.datetime.now().strftime('# creation_time="%I:%M%p %B %d %Y"\n')  # Tue Jan 26 13:57:06 CET 2016
        header+='# creation_time_epoch_ms="{}"\n'.format(int(time.time() * 1000.))
        header+='# username="{}"\n'.format(getpass.getuser())
        header+='# track_name="{}"\n'.format(car.track.name)
        header+='# car_name="{}"\n'.format(car.car_state.static_info.name)
        header+='# length_m="{}"\n'.format(car.car_state.static_info.length_m)
        header+='# width_m="{}"\n'.format(car.car_state.static_info.width_m)

        for m in self.csv_fields:
            if '.' in m:
                header+=m+',' # handle 'command.XXXX'
            else:
                v=getattr(self,m)
                if isinstance(v,Vector2): # handle vector fields
                    header+=m+'.x,'
                    header+=m+'.y,' # put x,y parts for Vector2
                else:
                    header+=m+',' # simple fields
        header=header[0:-1]  # chop trailing ,
        return header

    def parse_csv_row(self,row):
        """
        Sets all our fields contained in CSV file row. The column headers of row must match the field names in car_state exactly.
        :param row: DataFrame dict[key,value]
        :return: None
        """
        for f in self.csv_fields:
            try:
                v=row[f]
                if '.' in f: # e.g. command.steering, velocity_m_per_sec.x
                    parts=f.partition('.')
                    o=getattr(self,parts[0]) # e.g. self.command
                    if isinstance(getattr(o,parts[2]),bool):
                        v=True if v==1 else False
                    setattr(o,parts[2],v) # set self.command.steering to its column value
                else: # not an object with subfields, e.g. drift_angle_deg
                    if isinstance(getattr(self,f),bool):
                        v=True if v==1 else False
                    setattr(self,f,v)
            except KeyError:
                pass

    def get_record_csvrow(self):
            """

            :return: row of CSV file
            """
            l=[]
            for m in self.csv_fields:
                if '.' in m: # e.g. command.steering
                    parts=m.partition('.')
                    o=getattr(self,parts[0])
                    v=getattr(o,parts[2])
                else:
                    v=getattr(self,m)
                if isinstance(v,bool):
                    v=1 if v else 0
                    l.append(v)
                elif isinstance(v,Vector2):
                    l.append(v.x)
                    l.append(v.y)
                else:
                    l.append(v)
            s=''
            for v in l[:-1]:
                s=s+('{},'.format(v))
            s=s+('{:f}'.format(l[-1]))
            return s


