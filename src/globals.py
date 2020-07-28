""" global parameters"""
# DO NOT CHANGE UNLESS you change on server too
# define screen area, track is scaled to fill this area, note 4:3 aspect ratio
SCREEN_WIDTH_PIXELS=1024 #  pixels
SCREEN_HEIGHT_PIXELS= 768 # pixels
# meters per screen pixel, e.g. 4m car would be 40 pixels, so about 4% of width
# increase M_PER_PIXEL to make cars smaller relative to track
M_PER_PIXEL=0.15

SERVER_PING_INTERVAL_S=1 # interval between trying for server
SERVER_TIMEOUT_SEC=0.2 # timeout in seconds for UDP socket reads during game running
import scipy.constants
G=scipy.constants.value('standard acceleration of gravity')

#######################################################
# client

# SERVER_HOST='telluridevm.iniforum.ch' # metanet 16-core model server
SERVER_HOST='localhost'

FPS=30 # frames per second for simulation and animation
GAME_FONT_NAME='Consolas'
GAME_FONT_SIZE=16
# Joystick connectivity
CHECK_FOR_JOYSTICK_INTERVAL = 100 # check for missing joystick every this many cycles
JOYSTICK_NUMBER = 0 # todo in case multiple joysticks, use this to set the desired one, starts from zero

# recording data
DATA_FILENAME_BASE= 'l2race'
DATA_FOLDER_NAME= 'data'

# car and track options
CAR_NAME='l2racer' # label stuck on car
GAME_MODE='solo' # 'solo', 'multi', 'race' whether server launches track just for this client or puts car with others on same track
TRACK_NAME='track' # tracks are stored in the 'media' folder. Data for a track must be extracted using scripts in Track_Preparation before using in l2race
# track_name + '.png'
# track_name + '_map.npy'
# track_name + 'Info.npy'

#######################################################
#server and model settings. Client cannot affect these model server settings
SERVER_PORT = 50000 # client starts game on this port on the SERVER_HOST
CLIENT_PORT_RANGE='50010-50020' # range of ports used for client that server uses for game
DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK = True # set true for testing dynamics of car
FRICTION_FACTOR = .5 # overall friction parameter multiplier for some models
CLIENT_TIMEOUT_SEC=1 # server timeout in seconds before it ends thread for handling a car model
# client needs to open this port range for receiving state from server and sending commands to server


