# Driver control input from keyboard
from threading import Thread
from typing import Tuple

import easygui
import pygame
from pygame import KMOD_CTRL, KEYDOWN, KEYUP, K_QUESTION, K_h, KMOD_SHIFT, K_r, K_ESCAPE, K_y, K_l, K_o, K_w, K_t, K_p

from l2race_utils import my_logger
from l2race_settings import *

from tkinter import *
logger = my_logger(__name__)

from car_command import car_command
from user_input import user_input

def show_help():
    print(HELP)
    easygui.textbox("l2race help","l2race help",HELP)

def show_tk_help():
    root = Tk()
    root.withdraw()
    var = StringVar()
    label = Message( root, textvariable=var, relief=RAISED)
    var.set(HELP)
    label.pack()
    root.mainloop() # Todo windows does not show up sometimes


class my_keyboard:

    def __init__(self):
        """
        Makes a new my_keyboard

        :returns: new instance.

        """
        pygame.init()

        self.any_key_pressed=False # set true if any key is pressed
        self._auto_pressed=False # only used to log changes to/from autodrive
        self._run_user_model_pressed=False # only used to log changes to/from running user model
        self._restart_pressed=False # only used to log restarts and prevent multiple restarts being sent
        self._toggle_recording_pressed=False # only used to avoid multiple toggles of recording
        pygame.key.set_repeat(0) # disable repeat
        # pygame.key.set_repeat(int(1000./FPS))

    def read(self, car_command:car_command,user_input:user_input) -> None:
        """
        Read keyboard events and determines both car command and other types of UI commands.

        :param car_command: car command populated with user keyboard pressed down changes
        :param user_input: special commands that might be input

        """



        user_input.restart_car=False
        user_input.restart_client=False
        user_input.run_client_model=False
        user_input.toggle_recording=False
        user_input.open_playback_recording=False
        user_input.close_playback_recording=False
        user_input.choose_new_track=False
        user_input.toggle_paused=False

        for event in pygame.event.get(): # https://riptutorial.com/pygame/example/18046/event-loop
            if event.type == pygame.QUIT:
                user_input.quit = True
            elif event.type==KEYDOWN or event.type==KEYUP:
                key=event.key
                type=event.type
                mod=event.mod
                # definitely some type of keyboard event occured, something changed, process these types here
                if type==KEYUP:
                    # process key typed events
                    if key==K_QUESTION or key==K_h or key==pygame.K_F1:
                        show_help()
                    elif key==K_r:
                        if  mod&KMOD_SHIFT==0:
                            user_input.restart_car=True # r typed
                            logger.info(f'resetting car to start line')
                        else:
                            user_input.restart_client=True # R typed
                            logger.info(f'restarting client')
                    elif key==K_ESCAPE:
                        user_input.quit = True
                        logger.info(f'ESC key typed, quitting')
                    elif key==K_l:
                        user_input.toggle_recording= True
                    elif key==K_o and mod&KMOD_CTRL!=0:
                        user_input.open_playback_recording=True
                        logger.info(f'opening recording for playback')
                    elif key==K_w and mod&KMOD_CTRL!=0:
                        user_input.close_playback_recording=True
                        logger.info('closing playback')
                    elif key==K_y and mod&KMOD_SHIFT!=0:
                        car_command.autodrive_enabled=not car_command.autodrive_enabled
                        logger.info('disabled autodrive')
                    elif key==K_y and mod&KMOD_SHIFT==0:
                        car_command.autodrive_enabled=False
                        logger.info('disabled autodrive')
                    elif key==K_t:
                        user_input.choose_new_track=True
                        logger.info('user wants to select new track')
                    elif key==K_p:
                        user_input.toggle_paused=True
                        logger.info('user toggled pause')

                elif type==KEYDOWN:
                    if key==K_y and mod&KMOD_SHIFT==0: # k without shift held pushed down
                        car_command.autodrive_enabled=True
                        logger.info('enabled autodrive')


        # if no event, we still need to process control input by keys held pressed
        pressed = pygame.key.get_pressed()

        self.any_key_pressed=any(pressed)

        # following are key pressed down events that have effect while key is held pressed down
        if pressed[pygame.K_UP] or pressed[pygame.K_w]:
            car_command.throttle=1.
            car_command.brake=0.
        elif pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
            car_command.throttle=0.
            car_command.brake=1.

        if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
            car_command.steering = +1. # steer in negative angle direction, i.e. CW
        elif pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
            car_command.steering = -1. # steer in positive angle, i.e. CCW
        else:
            car_command.steering = 0.

        if pressed[pygame.K_SPACE]:
            car_command.reverse = True
        else:
            car_command.reverse = False
