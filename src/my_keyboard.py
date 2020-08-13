# Driver control input from keyboard
from typing import Tuple
import pygame

from src.l2race_utils import my_logger
logger = my_logger(__name__)

from src.car_command import car_command
from src.user_input import user_input

def printhelp():
    print('----------------------------\nKeyboard commands:\n'
          'drive with LEFT/UP/RIGHT/DOWN or AWDS keys\n'
          'hold SPACE pressed to reverse with drive keys\n'
          'y toggles automatic control (if implemented)\n'
          'r resets car\n'
          'R restarts client from scratch (if server went down)\n'
          'ESC quits\n'
          'h|? shows this help\n-----------------------------\n')
class my_keyboard:

    def __init__(self):
        pygame.init()
        self.car_command:car_command = car_command()
        self.user_input:user_input = user_input()
        printhelp()
        # self.backspace_down=False
        self.auto_pressed=False # only used to log changes to/from autodrive
        self.restart_pressed=False # only used to log restarts and prevent multiple restarts being sent

    def read(self) -> Tuple[car_command,user_input]:
        pressed = pygame.key.get_pressed()

        self.car_command.throttle = 0.
        self.car_command.brake = 0.
        self.car_command.steering = 0.
        self.user_input.restart_car=False
        self.user_input.restart_client=False

        # if not any(pressed):
        #     return self.car_input

        if pressed[pygame.K_y]:
            self.car_command.autodrive_enabled = True
            if not self.auto_pressed:
                logger.info('autodriver enabled')
                self.auto_pressed=True
        else:
            self.car_command.autodrive_enabled = False
            if self.auto_pressed:
                logger.info('autodriver disabled')
                self.auto_pressed=False

        if pressed[pygame.K_UP] or pressed[pygame.K_w]:
           self.car_command.throttle=1.
           self.car_command.brake=0.
        elif pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
           self.car_command.throttle=0.
           self.car_command.brake=1.
        elif pressed[pygame.K_SPACE]:
            pass

        if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
            self.car_command.steering = +1. # steer in negative angle direction, i.e. CW
        elif pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
            self.car_command.steering = -1. # steer in positive angle, i.e. CCW
        else:
            self.car_command.steering = 0.

        if pressed[pygame.K_SPACE]:
            self.car_command.reverse = True
        else:
            self.car_command.reverse = False

        if pressed[pygame.K_ESCAPE]:
            self.user_input.quit = True

        if pressed[pygame.K_r]:
            if not self.restart_pressed: # if it was not pushed before, then set flag true
                if (pygame.key.get_mods() & pygame.KMOD_LSHIFT):
                    self.user_input.restart_client =True
                else:
                    self.user_input.restart_car =True
                self.restart_pressed=True # mark that we set the flag
        else: # if r key not pressed anymore
                self.restart_pressed=False

        if pressed[pygame.K_QUESTION] or pressed[pygame.K_h]:
            printhelp()

        return self.car_command, self.user_input
