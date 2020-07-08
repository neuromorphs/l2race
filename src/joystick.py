"""
joystick race controller based on xbox bluetooth controller
"""
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0
from pygame import joystick

from src.car_input import car_input
from src.mylogger import mylogger

logger = mylogger(__name__)

class Joystick:

    def __init__(self):
        self.joy=None
        self.car_input=car_input()
        self.numAxes=None
        self.numButtons=None
        self.axes=None
        self.buttons=None

        pygame.init()
        joystick.init()
        count = joystick.get_count()
        if  count == 1:
            self.joy = joystick.Joystick(0)
            self.joy.init()
            self.numAxes = self.joy.get_numaxes()
            self.numButtons = self.joy.get_numbuttons()
            logger.info('joystick found with ' + str(self.numAxes) + ' axes and ' + str(self.numButtons) + ' buttons')
        else:
            logger.warning('no or too many joystick(s) found; only keyboard control possible')
            raise Exception('no joystick found')

    def read(self):
        pygame.event.get() # must call get() to handle internal queue
        # self.axes=list()
        # for i in range(self.numAxes):
        #     self.axes.append(self.joy.get_axis(i))
        #
        # self.buttons=list()
        # for i in range(self.numButtons):
        #     self.buttons.append(self.joy.get_button(i))

        self.car_input.steering = self.joy.get_axis(0) #self.axes[0]
        self.car_input.throttle = (1 + self.joy.get_axis(5)) / 2 # (1+self.axes[5])/2
        self.car_input.brake = (1+self.joy.get_axis(2))/2 #(1+self.axes[2])/2
        # print(self)
        return self.car_input

    def __str__(self):
        # axStr='axes: '
        # for a in self.axes:
        #     axStr=axStr+('{:5.2f} '.format(a))
        # butStr='buttons: '
        # for b in self.buttons:
        #     butStr=butStr+('1' if b else '_')
        #
        # return axStr+' '+butStr
        s="steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(self.car_input.steering, self.car_input.throttle, self.car_input.brake)
        return s


if __name__ == '__main__':
    joy=Joystick()
    while True:
        joy.read()
        pygame.time.wait(100)
        print("steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(joy.steering, joy.throttle, joy.brake))
        # print(str(joy))


