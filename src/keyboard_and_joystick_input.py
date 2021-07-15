"""
joystick race controller based on keyboard or xbox bluetooth controller.

"""
# import logging
from typing import Tuple
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0
from pygame import KEYDOWN, KEYUP

from my_joystick import my_joystick,printhelp as joystick_help
from my_keyboard import my_keyboard, show_help as keyboard_help

# from l2race_utils import my_logger
# logger = my_logger(__name__)
# logger.setLevel(logging.DEBUG)

from car_command import car_command
from user_input import user_input

def printhelp():
    joystick_help()
    keyboard_help()

class keyboard_and_joystick_input:
    """
    Input from either keyboard or joystick.
    """

    def __init__(self):
        self.joy=None
        self.keyboard=my_keyboard()

    def read(self, car_command:car_command,user_input:user_input):
        """
        Reads input from user from either keyboard or joystick. Keyboard takes precedence if any key is pressed.
        If user closes window, self.exit is set True.

        :param car_command: the command to fill in
        :param user_input: the user input (e.g. open file) to fill in
        """
        # Event queue
        if self.joy is None:
            try:
                self.joy=my_joystick()
            except RuntimeWarning:
                pass

        self.keyboard.read(car_command, user_input)  # definitely some type of keyboard event

        if self.keyboard.any_key_pressed:
            # if any key is pressed down, don't use joystick input
            return

        # if no keyboard event but we have a joystick, get command and user input from that
        if self.joy is not None :
            try:
                self.joy.read(car_command, user_input)
                return
            except:
                self.joy=None

        return  # no effect on command



