"""
joystick race controller based on xbox bluetooth controller.
Xbox has 11 buttons.
buttons ABXY are first four buttons 0-3, then menu and window buttons are 4th and 5th from end, i.e. 7 and 6
"""
# import logging
from typing import Tuple
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0

from src.my_joystick import my_joystick,printhelp as joystick_help
from src.my_keyboard import my_keyboard, show_help as keyboard_help

# from src.l2race_utils import my_logger
# logger = my_logger(__name__)
# logger.setLevel(logging.DEBUG)

from src.car_command import car_command
from src.user_input import user_input

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
        self.exit=False

    def read(self) -> Tuple[car_command,user_input]:
        """
        Reads input from user from either keyboard or joystick. Keyboard takes precedence if any key is pressed.
        If user closes window, self.exit is set True.

        :returns: the car_command,user_input
        """
        # Event queue
        if self.joy is None:
            try:
                self.joy=my_joystick()
            except RuntimeWarning:
                pass

        for event in pygame.event.get(): # https://riptutorial.com/pygame/example/18046/event-loop
            if event.type == pygame.QUIT:
                self.exit = True

        kc,ku=self.keyboard.read()

        self.exit=ku.quit

        if self.keyboard.any_key_pressed:
            return kc,ku # start with keyboard input

        if self.joy :
            try:
                jc,ju=self.joy.read()
                self.exit=ju.quit
                return jc,ju
            except:
                self.joy=None
                return kc,ku

        return kc,ku

    def __str__(self):
        # axStr='axes: '
        # for a in self.axes:
        #     axStr=axStr+('{:5.2f} '.format(a))
        # butStr='buttons: '
        # for b in self.buttons:
        #     butStr=butStr+('1' if b else '_')
        #
        # return axStr+' '+butStr
        s="steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(self.car_command.steering, self.car_command.throttle, self.car_command.brake)
        return s


