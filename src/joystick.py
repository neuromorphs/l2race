"""
joystick race controller based on xbox bluetooth controller.
Xbox has 11 buttons.
buttons ABXY are first four buttons 0-3, then menu and window buttons are 4th and 5th from end, i.e. 7 and 6
"""
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0
from pygame import joystick

from src.car_input import car_input
from src.mylogger import mylogger

logger = mylogger(__name__)

def printhelp():
    print('Joystick commands:\n'
          'steer with left joystick left|right\n'
          'throttle is right paddle, brake is left paddle\n'
          'B toggles reverse gear\n'
          'Menu button resets car\n'
          'Windows button quits\n'
          )


class Joystick:

    def __init__(self):
        self.joy=None
        self.car_input=car_input()
        self.numAxes=None
        self.numButtons=None
        self.axes=None
        self.buttons=None

        self._rev_was_pressed=False # to go to reverse mode or toggle out of it

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
        printhelp()

    def read(self):
        pygame.event.get() # must call get() to handle internal queue

        self.car_input.steering = self.joy.get_axis(0) #self.axes[0], returns + for right push, which should make steering angle positive, i.e. CW
        self.car_input.throttle = (1 + self.joy.get_axis(5)) / 2 # (1+self.axes[5])/2
        self.car_input.brake = (1+self.joy.get_axis(2))/2 #(1+self.axes[2])/2
        self.car_input.reset=self.joy.get_button(7) # menu button
        self.car_input.quit=self.joy.get_button(6) # windows button
        revPressed=self.joy.get_button(1) # B button
        if revPressed and not self._rev_was_pressed: # if it was not pressed last time and is pressed now, toggle reverse
            self.car_input.reverse=not self.car_input.reverse
        self._rev_was_pressed=revPressed

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
    it=0
    while True:
        # joy.read()
        # print("steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(joy.steering, joy.throttle, joy.brake))
        pygame.event.get() # must call get() to handle internal queue
        joy.axes=list()
        for i in range(joy.numAxes):
            joy.axes.append(joy.joy.get_axis(i))

        joy.buttons=list()
        for i in range(joy.numButtons):
            joy.buttons.append(joy.joy.get_button(i))
        axStr='axes: '
        for i in joy.axes:
            axStr=axStr+('{:5.2f} '.format(i))
        butStr='buttons: '
        for i in joy.buttons:
            butStr=butStr+('1' if i else '_')


        print(str(it)+': '+axStr+' '+butStr)
        it+=1
        pygame.time.wait(300)


