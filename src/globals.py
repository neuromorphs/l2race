""" global parameters"""
# all distances are in units of pixels on the playing surface, which we take as meters
# i.e. velocity is pix/s, acceleration is pix/s/s etc

# TODO if car is 4m long, then that would be 4 pixels with M_PER_PIXEL=1, which is very tiny on screen
import scipy.constants

SERVER_PORT = 50000

# define screen area, track is scaled to fill this area, note 4:3 aspect ratio
SCREEN_WIDTH_PIXELS=1024 #  pixels
SCREEN_HEIGHT_PIXELS= 768 # pixels

M_PER_PIXEL=0.1 # meters per screen pixel, e.g. 4m car would be 40 pixels, so about 4% of width
SOCKET_TIMEOUT_SEC=1 # timeout for UDP socket reads

G=scipy.constants.value('standard acceleration of gravity')
