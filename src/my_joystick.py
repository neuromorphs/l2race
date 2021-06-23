"""
joystick race controller based on xbox bluetooth controller.
Xbox has 11 buttons.
buttons ABXY are first four buttons 0-3, then menu and window buttons are 4th and 5th from end, i.e. 7 and 6
"""
import logging
from typing import Tuple, Optional
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0
from pygame import joystick
import platform
import time

INACTIVITY_RECONNECT_TIME = 15
RECONNECT_TIMEOUT = 5

from src.l2race_utils import my_logger
from src.globals import JOYSTICK_NUMBER
logger = my_logger(__name__)
# logger.setLevel(logging.DEBUG)

from src.car_command import car_command
from src.user_input import user_input

def printhelp():
    print('\n-----------------------------------\nJoystick commands:\n'
          'steer with left joystick left|right\n'
          'throttle is right paddle, brake is left paddle\n'
          'B activates reverse gear\n'
          'Y activates autordrive control (if implemented)\n'
          'A runs the client ghost car model\n'
          'Menu button (to left of XYBA buttons) resets car\n'
          'X restarts client from scratch (if server went down)\n'
          'Windows button (by left joystick) quits\n-----------------------------------\n'
          )


class my_joystick:
    """"
    The read() method gets joystick input to return (car_command, user_input)
    """
    XBOX_ONE_BLUETOOTH_JOYSTICK = 'Xbox One S Controller' # XBox One when connected as Bluetooth in Windows
    XBOX_WIRED = 'Controller (Xbox One For Windows)' #Although in the name the word 'Wireless' appears, the controller is wired
    XBOX_ELITE = 'Xbox One Elite Controller' # XBox One when plugged into USB in Windows
    PS4_DUALSHOCK4 = 'Sony Interactive Entertainment Wireless Controller' # Only wired connection tested
    PS4_WIRELESS_CONTROLLER = 'Sony Computer Entertainment Wireless Controller' # Only wired connection on macOS tested

    def __init__(self, joystick_number=JOYSTICK_NUMBER):
        """
        Makes a new joystick instance.

        :param joystick_number: use if there is more than one joystick
        :returns: the instance

        :raises RuntimeWarning if no joystick is found or it is unknown type
        """
        self.joy: Optional[joystick.Joystick] = None
        self.car_command: car_command = car_command()
        self.user_input: user_input = user_input()
        self.default_car_command = car_command()
        self.default_user_input = user_input()
        self.numAxes: Optional[int] = None
        self.numButtons: Optional[int] = None
        self.axes = None
        self.buttons = None
        self.name: Optional[str] = None
        self.joystick_number: int = joystick_number
        self.lastTime = 0
        self.lastActive = 0
        self.run_user_model_pressed = False  # only used to log changes to/from running user model

        self._rev_was_pressed = False  # to go to reverse mode or toggle out of it
        joystick.init()
        count = joystick.get_count()
        if count < 1:
            raise RuntimeWarning('no joystick(s) found')

        self.platform = platform.system()

        self.lastActive=time.time()
        try:
            self.joy = joystick.Joystick(self.joystick_number)
        except:
            if self.platform == 'Linux': # Linux list joysticks from 3 to 0 instead of 0 to 3
                if self.joystick_number == 0:
                    self.joy = joystick.Joystick(3)
                else:
                    self.joy = joystick.Joystick(4 - self.joystick_number)

        self.joy.init()
        self.numAxes = self.joy.get_numaxes()
        self.numButtons = self.joy.get_numbuttons()
        self.name=self.joy.get_name()
        logger.info(f'joystick is named "{self.name}"')
        if not self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK \
                and not self.name == my_joystick.XBOX_ELITE \
                and not self.name == my_joystick.XBOX_WIRED \
                and not self.name == my_joystick.PS4_WIRELESS_CONTROLLER \
                and not self.name == my_joystick.PS4_DUALSHOCK4:
            logger.warning('Name: {}'.format(self.name))
            logger.warning('Unknown joystick type {} found.'
                           'Add code to correctly map inputs by running my_joystick as main'.format(self.name))
            raise RuntimeWarning('unknown joystick type "{}" found'.format(self.name))
        logger.debug('joystick named "{}" found with {} axes and {} buttons'.format(self.name, self.numAxes, self.numButtons))

    def _connect(self):
        """
        Tries to connect to joystick

        :return: None

        :raises RuntimeWarning if there is no joystick
        """


    def check_if_connected(self):
        """
        Periodically checks if joystick is still there if there has not been any input for a while.
        If we have not read any input for INACTIVITY_RECONNECT_TIME and we have not checked for RECONNECT_TIMEOUT, then quit() the joystick and reconstruct it.

        :raises RuntimeWarning if check fails
        """
        now = time.time()
        if now - self.lastActive > INACTIVITY_RECONNECT_TIME and now - self.lastTime > RECONNECT_TIMEOUT:
            self.lastTime = now
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                return
            else:
               raise RuntimeWarning('no joystick found')


    def read(self) -> Tuple[car_command,user_input]:
        """
        Returns the car_command, user_input tuple. Use check_if_connected() and connect() to check and connect to joystick.

        :returns: the car_command,user_input

        :raises RuntimeWarning if joystick disappears
        """

        self.check_if_connected()

        if self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK and self.platform == 'Darwin':
            # Buttons A B X Y
            A = 0  # ghost (client_model)
            B = 1  # reverse
            X = 3  # restart game
            Y = 4  # autodrive
            # Quit and Restart Client buttons
            Quit = 16  # quit client
            Restart_Client = 11  # restart client
            # Analog Buttons and Axes
            Steering = 0
            Throttle = 4
            Brake = 5

        elif self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK:
            A = 0  # ghost (client_model)
            B = 1  # reverse
            X = 2  # restart game
            Y = 3  # autodrive
            # Quit and Restart Client buttons
            Quit = 6  # quit client
            Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            Steering = 0
            Throttle = 5
            Brake = 4

        elif self.name == my_joystick.XBOX_ELITE: # antonio's older joystick? also XBox One when plugged into USB cable on windows
            # Buttons A B X Y
            A = 0  # ghost (client_model)
            B = 1  # reverse
            X = 2  # restart game
            Y = 3  # autodrive
            # Quit and Restart Client buttons
            Quit = 6  # quit client
            Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            Steering = 0
            Throttle = 5
            Brake = 2

        elif self.name == my_joystick.XBOX_WIRED:
            # Buttons A B X Y
            A = 0  # ghost (client_model)
            B = 1  # reverse
            X = 2  # restart game
            Y = 3  # autodrive
            # Quit and Restart Client buttons
            Quit = 6  # quit client
            Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            Steering = 0
            Throttle = 5
            Brake = 4

        elif self.name == my_joystick.PS4_DUALSHOCK4:
            # Buttons X O Square Triangle
            X = 0  # ghost (client_model)
            O = 1  # reverse
            Square = 3  # restart game
            Triangle = 2  # autodrive
            # Quit and Restart Client buttons
            Quit = 10
            Restart_Client = 3
            # Analog Buttons and Axes
            Steering = 0
            Throttle = 5
            Brake = 2
            
        elif self.name == my_joystick.PS4_WIRELESS_CONTROLLER:
            # Buttons X O Square Triangle
            X = 1  # ghost (client_model)
            O = 2  # reverse
            Square = 0  # restart game
            Triangle = 3  # autodrive
            # Quit and Restart Client buttons
            Quit = 8
            Restart_Client = 9
            # Analog Buttons and Axes
            Steering = 2
            Throttle = 4
            Brake = 3

            
            
        if 'Xbox' in self.name:
            # Buttons A B X Y
            self.user_input.restart_client = self.joy.get_button(X)  # X button - restart
            self.car_command.reverse = True if self.joy.get_button(B) == 1 else False  # B button - reverse
            if not self.car_command.reverse:  # only if not in reverse
                self.car_command.autodrive_enabled = True if self.joy.get_button(Y) == 1 else False  # Y button - autodrive
            self.user_input.run_client_model = self.joy.get_button(A)  # A button - ghost

        elif 'Sony' in self.name:
            # Buttons X O Square Triangle
            self.user_input.restart_client = self.joy.get_button(Square)  # Square button - restart
            self.car_command.reverse = True if self.joy.get_button(O) == 1 else False  # O button - reverse
            if not self.car_command.reverse:  # only if not in reverse
                self.car_command.autodrive_enabled = True if self.joy.get_button(Triangle) == 1 else False  # Triangle button - autodrive
            self.user_input.run_client_model = self.joy.get_button(X)  # X button - ghost

        # Quit and Restart Client buttons
        self.user_input.restart_car = self.joy.get_button(Restart_Client)  # menu button - Restart Client
        self.user_input.quit = self.joy.get_button(Quit)  # windows button - Quit Client

        # Analog Buttons and Axes
        self.car_command.steering = self.joy.get_axis(
            Steering)  # Steering returns + for right push, which should make steering angle positive, i.e. CW
        self.car_command.throttle = (1 + self.joy.get_axis(Throttle)) / 2.  # Throttle
        self.car_command.brake = (1 + self.joy.get_axis(Brake)) / 2.  # Brake



        logger.debug(self)
        return self.car_command, self.user_input

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


if __name__ == '__main__':
    from colorama import init, deinit, Fore, Style
    import numpy as np
    import atexit
    init()
    atexit.register(deinit)
    pygame.init()
    joy = my_joystick()
    new_axes = np.zeros(joy.numAxes,dtype=float)
    old_axes = new_axes.copy()
    it = 0
    print('Name of the current joystick is {}.'.format(joy.name))
    print('Your platform name is {}'.format(joy.platform))
    while True:
        # joy.read()
        # print("steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(joy.steering, joy.throttle, joy.brake))
        pygame.event.get()  # must call get() to handle internal queue
        for i in range(joy.numAxes):
            new_axes[i] = joy.joy.get_axis(i)  # assemble list of analog values
        diff = new_axes-old_axes
        old_axes = new_axes.copy()
        joy.buttons = list()
        for i in range(joy.numButtons):
            joy.buttons.append(joy.joy.get_button(i))

        # format output so changed are red
        axStr = 'axes: '
        axStrIdx = 'axes:'
        for i in range(joy.numAxes):
            if abs(diff[i]) > 0.3:
                axStr += (Fore.RED+Style.BRIGHT+'{:5.2f} '.format(new_axes[i])+Fore.RESET+Style.DIM)
                axStrIdx += '__'+str(i)+'__'
            else:
                axStr += (Fore.RESET+Style.DIM+'{:5.2f} '.format(new_axes[i]))
                axStrIdx += '______'

        butStr = 'buttons: '
        for button_index, button_on in enumerate(joy.buttons):
            butStr = butStr+(str(button_index) if button_on else '_')

        print(str(it) + ': ' + axStr + ' '+butStr)
        print(str(it) + ': ' + axStrIdx)
        it += 1
        pygame.time.wait(300)


