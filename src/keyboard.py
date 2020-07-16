# User input
import pygame

from src.car_input import car_input
from src.mylogger import mylogger
logger = mylogger(__name__)

class Keyboard:

    def __init__(self):
        pygame.init()
        self.car_input = car_input()

    def read(self):
        pressed = pygame.key.get_pressed()

        self.car_input.throttle = 0
        self.car_input.brake = 0
        self.car_input.steering = 0

        if not any(pressed):
            return self.car_input

        if pressed[pygame.K_UP] or pressed[pygame.K_w]:
           self.car_input.throttle=1
           self.car_input.brake=0
        elif pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
           self.car_input.throttle=0
           self.car_input.brake=1
        elif pressed[pygame.K_SPACE]:
            pass

        if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
            self.car_input.steering =1
        elif pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
            self.car_input.steering =-1
        elif pressed[pygame.K_BACKSPACE]:
            self.car_input.reverse != self.car_input.reverse
        elif pressed[pygame.K_ESCAPE]:
            self.car_input.quit =True
        elif pressed[pygame.K_r]:
            self.car_input.reset =True
        else:
            self.car_input.steering = 0
        return self.car_input
