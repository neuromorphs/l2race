""" global parameters"""
import logging


LOGGING_LEVEL=logging.INFO # set the overall default leval, change with --log option

import scipy.constants
G=scipy.constants.value('standard acceleration of gravity')

#######################################################
# client

# SERVER_HOST='telluridevm.iniforum.ch' # metanet 16-core model server
SERVER_HOST='localhost'
SERVER_PING_INTERVAL_S = 1  # interval between trying for server
SERVER_TIMEOUT_SEC = 1  # timeout in seconds for UDP socket reads during game running
ENABLE_UPNP = True  # set True to try unpnp to forward CLIENT_PORT_RANGE ports to local machine
UPNP_LEASE_TIME = 1200  # the lease time for these ports in seconds

# your autodrive controller module (i.e. folder) and class name, must be a class that has read method that returns the car_command() object
# AUTODRIVE_MODULE='src.controllers.pid_next_waypoint_car_controller'
# AUTODRIVE_CLASS='pid_next_waypoint_car_controller'
# overridden by command line --autodrive
# AUTODRIVE_MODULE='src.controllers.pure_pursuit_controller'
# AUTODRIVE_CLASS = 'pure_pursuit_controller'
AUTODRIVE_MODULE='src.controllers.pure_pursuit_controller_v2'
AUTODRIVE_CLASS = 'pure_pursuit_controller_v2'

# your model class that takes car state and control and predicts the next state given a future time.
# overridden by command line --model
CAR_MODEL_MODULE= 'src.models.models' # the module (i.e. folder.file without .py)
CAR_MODEL_CLASS = 'linear_extrapolation_model' # the class within the file
# CAR_MODEL_CLASS= 'RNN_model'

#display
FPS=20 # frames per second for simulation and animation
GAME_FONT_NAME = 'Consolas'  # local display font, default is Consolas
GAME_FONT_SIZE = 16  # default is 16

# Joystick connectivity
CHECK_FOR_JOYSTICK_INTERVAL = 100 # check for missing joystick every this many cycles
JOYSTICK_NUMBER = 0 # in case multiple joysticks, use this to set the desired one, starts from zero

# recording data
DATA_FILENAME_BASE= 'l2race'
DATA_FOLDER_NAME= 'data'

# car and track options
CAR_NAME='l2racer' # label stuck on car
TRACK_NAME='oval_easy' # tracks are stored in the 'media' folder. Data for a track must be extracted using scripts in Track_Preparation before using in l2race
TRACKS_FOLDER='./media/tracks/' # location of tracks relative to root of l2race
# Other possible track names:
# track_names = [
#          'Sebring',
#          'oval',
#          'oval_easy',
#          'track_1',
#          'track_2',
#          'track_3',
#          'track_4',
#          'track_5',
#          'track_6']
# track_name + '.png'
# track_name + '_map.npy'
# track_name + 'Info.npy'

#######################################################
# server and model settings. Client cannot affect these model server settings
#
# DO NOT CHANGE THESE VALUES
#########################
# DO NOT CHANGE UNLESS they are also changed on model server
# Define screen area, track is scaled to fill this area, note 4:3 aspect ratio
# Track information must also be generated at this size so client cannot change the values easily.
SCREEN_WIDTH_PIXELS = 1024  # pixels
SCREEN_HEIGHT_PIXELS = 768  # pixels
# meters per screen pixel, e.g. 4m car would be 40 pixels, so about 4% of width
# increase M_PER_PIXEL to make cars smaller relative to track
M_PER_PIXEL = 0.10  # 0.2 makes the cars really small on track. 0.1 makes them fill about 1/3 of track width.

SERVER_PORT = 50000  # client starts game on this port on the SERVER_HOST
CLIENT_PORT_RANGE = '50010-50020'  # range of ports used for client that server uses for game
    # client needs to open/forward this port range for receiving state from server and sending commands to server
    # The ENABLE_UPNP flag turns on automatic forwarding but it does not work with all routers.
KILL_ZOMBIE_TRACK_TIMEOUT_S = 10  # if track process gets no input for this long, it terminates itself
FRICTION_FACTOR = .5  # overall friction parameter multiplier for some models
SAND_SLOWDOWN = 0.975  # If in sand, at every update the resulting velocity is multiplied by the slowdown factor
REVERSE_TO_FORWARD_GEAR = 0.5  # You get less acceleration on reverse gear than while moving forwards.
MODEL_UPDATE_RATE_HZ = 20  # rate that server attempts to update all the car models for each track process (models run serially in each track process)
MAX_CARS_PER_TRACK = 6  # only this many cars can run on each track
MAX_SPECTATORS_PER_TRACK = 10  # only this many spectators can connect to each track
KS_TO_ST_SPEED_M_PER_SEC = 2.0  # transistion speed from KS to ST model types


### Constants for RNN0 model:
import pandas as pd
import numpy as np

normalization_distance = M_PER_PIXEL*(np.sqrt((SCREEN_HEIGHT_PIXELS ** 2) + (SCREEN_WIDTH_PIXELS ** 2)))
normalization_velocity = 50.0  # from Mark 24
normalization_acceleration = 5.0  # 2.823157895
normalization_angle = 180.0
normalization_dt = 1.0e-1
normalization_x = SCREEN_WIDTH_PIXELS
normalization_y = SCREEN_HEIGHT_PIXELS

NORMALIZATION_INFO = pd.DataFrame({
    'time': None,
    'dt': normalization_dt,
    'command.autodrive_enabled': None,
    'command.steering': None,
    'command.throttle': None,
    'command.brake': None,
    'command.reverse': None,
    'position_m.x': normalization_distance,
    'position_m.y': normalization_distance,
    'velocity_m_per_sec.x': normalization_velocity,
    'velocity_m_per_sec.y': normalization_velocity,
    'speed_m_per_sec': normalization_velocity,
    'accel_m_per_sec_2.x': normalization_acceleration,
    'accel_m_per_sec_2.y': normalization_acceleration,
    'steering_angle_deg': None,
    'body_angle_deg': normalization_angle,
    'body_angle.cos': None,
    'body_angle.sin': None,
    'yaw_rate_deg_per_sec': None,
    'drift_angle_deg': None,
    'hit_distance': normalization_distance,
    'nearest_waypoint_idx': None,
    'first_next_waypoint.x': normalization_distance,
    'first_next_waypoint.y': normalization_distance,
    'fifth_next_waypoint.x': normalization_distance,
    'fifth_next_waypoint.y': normalization_distance,
    'twentieth_next_waypoint.x': normalization_distance,
    'twentieth_next_waypoint.y': normalization_distance
}, index=[0])

