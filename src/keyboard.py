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

        if not any(pressed):
            return self.car_input

        if pressed[pygame.K_UP]:
           self.car_input.throttle=1
           self.car_input.brake=0
        elif pressed[pygame.K_DOWN]:
           self.car_input.throttle=0
           self.car_input.brake=1
        elif pressed[pygame.K_SPACE]:
            pass
        if pressed[pygame.K_RIGHT]:
            self.car_input.steering =-1
        elif pressed[pygame.K_LEFT]:
            self.car_input.steering =1
        else:
            self.car_input.steering = 0
        return self.car_input
