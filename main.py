import importlib

from src.globals import AUTODRIVE_MODULE, AUTODRIVE_CLASS
from src.client import define_game
from src.l2race_utils import my_logger
logger=my_logger(__name__)

# main class for clients that run a car on the l2race track.

if __name__ == '__main__':

    '''
    Here is place for your code to specify
    what arguments you wish to pass to the game instance
    e.g:
    
    # track_names = ['Sebring',
    #          'oval',
    #          'track_1',
    #          'track_2',
    #          'track_3',
    #          'track_4',
    #          'track_5',
    #          'track_6']
    #
    # import random
    # track_name = random.choice(track_names)
    
    '''

    '''
    define_game accepts various arguments which specify the game you want to play
    These arguments are:
    gui:            'with_gui'/'without gui',
    track_name:     'Sebring','oval','track_1','track_2','track_3','track_4','track_5','track_6'
    car_name:       any string
    server_host:    [change by user not recommended]
    server_port:    [change by user not recommended]
    joystick_number: [change only in multi-player game os the same device]?
    fps             [change by user not recommended]
    timeout_s       [change by user not recommended]
    record          True/False
    
    Providing arguments to define_game function is optional
    If an argument is not provided below the program checks if it was provided as corresponding flag.
    If also no corresponding flag was provided the program takes a default value
    The only case when a flag has precedence over a variable provided below is for disabling gui
    '''

    try:
        i = importlib.import_module(AUTODRIVE_MODULE)
        c=getattr(i, AUTODRIVE_CLASS)
        controller = c() # set it to a class in globals.py
    except Exception as e:
        logger.error('cannot import AUTODRIVE_CLASS named {} from module AUTODRIVE_MODULE named {}, got {}'.format(AUTODRIVE_CLASS, AUTODRIVE_MODULE,e))
        controller=None


    game = define_game(gui=False,
                       ctrl=controller)
    game.run()
    # game.replay('l2race-20200803-220912')
    # game.replay() # todo put in client as argument to client

    '''
    Place for your code to post-process data
    '''
