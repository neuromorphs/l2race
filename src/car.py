import os

import pygame
from pygame.math import Vector2
from math import sin, cos, radians, degrees, copysign
from src.mylogger import mylogger
from src.track import Track

logger = mylogger(__name__)
from src.carstate import CarState
from src.globals import SCREEN_WIDTH, SCREEN_HEIGHT


import sys
sys.path.append('../commonroad-vehicle-models/Python')
from parameters_vehicle1 import parameters_vehicle1
from init_KS import init_KS
from init_ST import init_ST
from init_MB import init_MB
from vehicleDynamics_KS import vehicleDynamics_KS
from vehicleDynamics_ST import vehicleDynamics_ST
from vehicleDynamics_MB import vehicleDynamics_MB

class Car:
    """
    Model of car including dynamical model.
    """


    def __init__(self, x=500, y=500):
        self.car_state = CarState(x, y)
        self.track=Track() # TODO for now just use default track
        self.closestTrackVertex=None
        # TODO change color of car to be unique, add name of car
        self.name = 'car'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "../media/"+self.name+".png")
        self.image = pygame.image.load(image_path)  # load image of car

        # scale it to its length in pixels (all units are in pixels which are meters)
        rect = self.image.get_rect()
        sc = self.car_state.length/rect.width
        self.image = pygame.transform.scale(self.image, (int(sc * rect.width), int(sc * rect.height)))

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.image, self.car_state.angle_deg)
        rect = rotated.get_rect()
        screen.blit(rotated, ((self.car_state.position) - (int(rect.width / 2), int(rect.height / 2))))

    def update(self, dt, input):

        if input.reset:
            logger.info('resetting car')
            self.reset()

        self.car_state.speed =self.car_state.velocity.length()  # in case something external touched component of velocity, like collision
        acceleration=input.throttle*self.car_state.max_acceleration-input.brake * self.car_state.brake_deceleration
        if input.reverse: # todo fix, logic incorrect
            acceleration=-acceleration

        if acceleration==0:
            acceleration=-self.car_state.free_deceleration

        dacc=acceleration * dt
        self.car_state.speed += dacc
        if self.car_state.speed<0:
            self.car_state.speed=0
        elif self.car_state.speed>self.car_state.max_speed:
            self.car_state.speed=self.car_state.max_speed

        self.car_state.steering=-input.steering*(self.car_state.max_steering*((2*self.car_state.max_speed-self.car_state.speed)/(2*self.car_state.max_speed)))
        if not self.car_state.steering==0:
            turning_radius = self.car_state.length / sin(radians(self.car_state.steering))
            angular_velocity = self.car_state.speed / turning_radius
        else:
            angular_velocity=0
        self.car_state.angle_deg += degrees(angular_velocity) * dt

        self.car_state.velocity=Vector2(self.car_state.speed * cos(radians(self.car_state.angle_deg)), -self.car_state.speed * sin(radians(self.car_state.angle_deg)))

        self.car_state.position += self.car_state.velocity * dt

        self.locate()

        w = SCREEN_WIDTH
        h = SCREEN_HEIGHT
        if self.car_state.position.x > w:
            self.car_state.position.x = w
            self.car_state.velocity.x = -self.car_state.velocity.x
        elif self.car_state.position.x < 0:
            self.car_state.position.x = 0
            self.car_state.velocity.x = -self.car_state.velocity.x
        if self.car_state.position.y > h:
            self.car_state.position.y = h
            self.car_state.velocity.y = -self.car_state.velocity.y
        elif self.car_state.position.y < 0:
            self.car_state.position.y = 0
            self.car_state.velocity.y = -self.car_state.velocity.y
        # logger.info(self.car_state)


    def locate(self):
        """ locates car on track and updates in the car_state"""
        # vs=self.track.vertices
        # minDist=None if self.closestTrackVertex==None else minDist=(self.closestTrackVertex-Vector2(vs[]))
        # for p in :

        pass


    def reset(self):
        """ reset car to starting position"""
        x=0
        y=0
        if self.track:
            x=self.track.vertices[0][0]
            y=self.track.vertices[0][0]
        self.car_state.position = Vector2(x,y) # todo reset to start line of track