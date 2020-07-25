""" global parameters"""

SERVER_PORT = 50000
# SERVER_HOST='192.168.0.206' # tobi home internal lan ip
# SERVER_HOST='178.82.113.207' # tobi home external lan ip
SERVER_HOST='localhost'

# DO NOT CHANGE UNLESS you change on server too
# define screen area, track is scaled to fill this area, note 4:3 aspect ratio
SCREEN_WIDTH_PIXELS=1024 #  pixels
SCREEN_HEIGHT_PIXELS= 768 # pixels
# meters per screen pixel, e.g. 4m car would be 40 pixels, so about 4% of width
# increase M_PER_PIXEL to make cars smaller relative to track
M_PER_PIXEL=0.15

SERVER_PING_INTERVAL_S=1 # interval between trying for server
SOCKET_TIMEOUT_SEC=0.2 # timeout for UDP socket reads
import scipy.constants
G=scipy.constants.value('standard acceleration of gravity')

# client
FPS=30 # frames per second for simulation and animation
GAME_FONT_NAME='Consolas'
GAME_FONT_SIZE=16
# Joystick connectivity
CHECK_FOR_JOYSTICK_INTERVAL = 100 # check for missing joystick every this many cycles
JOYSTICK_NUMBER = 0 # todo in case multiple joysticks, use this to set the desired one, starts from zero

#server and model settings. Client cannot affect these model server settings
DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK = True # set true for testing dynamics of car
FRICTION_FACTOR = .5 # overall friction parameter multiplier for some models


