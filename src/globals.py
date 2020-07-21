""" global parameters"""
# all distances are in units of pixels on the playing surface, which we take as meters
# i.e. velocity is pix/s, acceleration is pix/s/s etc

# TODO if car is 4m long, then that would be 4 pixels with M_PER_PIXEL=1, which is very tiny on screen
import scipy.constants

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

SOCKET_TIMEOUT_SEC=.1 # timeout for UDP socket reads
G=scipy.constants.value('standard acceleration of gravity')

# client
FPS=30 # frames per second for simulation and animation
CHECK_FOR_JOYSTICK_INTERVAL = 100 # check for missing joystick every this many cycles
GAME_FONT_NAME='Consolas'
GAME_FONT_SIZE=16

#server
DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK=True

DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK = True

# Joystick connectivity
WIRELESS = False
JOY_NUMBER = 2
