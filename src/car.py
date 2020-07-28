# class for Car, holds other important stuff
import os
from math import radians, cos, sin

import pygame
from pygame.math import Vector2
from src.my_logger import my_logger
from src.track import track

from src.car_state import car_state
from src.globals import M_PER_PIXEL, G, CAR_NAME, GAME_FONT_NAME, GAME_FONT_SIZE

logger = my_logger(__name__)


class car:
    """
    Local model of car. It has CarState() that is updated by remote server, and methods for drawing car and other static information related to car that is not transmitted over socket.
    """

    def __init__(self,name=CAR_NAME, image_name='car_1'): # TODO initialize at starting line with correct angle; this is job of model server
        self.car_state = car_state() # TODO change to init to starting line of track
        self.track=track # TODO for now just use default track TODO check if Track should be field of Car()
        # TODO change color of car to be unique, add name of car
        self.image_name = image_name # TODO make part of constructor?
        self.loadAndScaleCarImage()
        self.name=name
        self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        # self.rect = self.image.get_rect()

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.image, -self.car_state.body_angle_deg)
        rect = rotated.get_rect()
        screen.blit(rotated, ((self.car_state.position_m/M_PER_PIXEL) - (int(rect.width / 2), int(rect.height / 2))))
        # label name
        self.game_font.render_to(screen, (self.car_state.position_m.x/M_PER_PIXEL, self.car_state.position_m.y/M_PER_PIXEL), self.name, [200,200,200]),

        # draw acceleration
        len=(self.car_state.accel_m_per_sec_2.x/G)*(self.car_state.length_m * 6) # self.car_state.command.throttle*self.car_state.length*2 # todo fix when accel include lateral component
        body_rad=radians(self.car_state.body_angle_deg)
        body_vec=(len*cos(body_rad),len*sin(body_rad))
        pygame.draw.line(screen, [255,50,50],self.car_state.position_m/M_PER_PIXEL, (self.car_state.position_m+body_vec)/M_PER_PIXEL,1)

        # draw steering command
        str_len= self.car_state.length_m / 2
        str_orig=self.car_state.position_m+(cos(body_rad)*str_len,sin(body_rad)*str_len)
        str_rad=radians(self.car_state.body_angle_deg+self.car_state.steering_angle_deg)
        str_vec=(str_len*cos(str_rad),str_len*sin(str_rad))
        str_pos1=str_orig-str_vec
        str_pos2=str_orig+str_vec
        pygame.draw.line(screen, [50,250,250],str_pos1/M_PER_PIXEL, str_pos2/M_PER_PIXEL,2)


    def loadAndScaleCarImage(self):
        """ loads image for car and scales it to its actual length.
        Call only after car_state is filled by server
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "../media/" + self.image_name + ".png") # todo name of car and file should come from server
        self.image = pygame.image.load(image_path)  # load image of car

        # scale it to its length in pixels (all units are in pixels which are meters)
        # TODO use global scale of M_PER_PIXEL correctly here
        rect = self.image.get_rect()
        sc = self.car_state.length_m / (M_PER_PIXEL * rect.width)
        self.image = pygame.transform.scale(self.image, (int(sc * rect.width), int(sc * rect.height)))

    # I think we do not use this function. Almost sure but tired today so I leave it commented
    # def reset(self): # TODO (Marcin): Is this function used at all? If not delete. track has not attribute vertices..
    #     """ reset car to starting position"""
    #     self.car_state.reset()
    #     x = 0
    #     y = 0
    #     if self.track:
    #         x = self.track.vertices[0][0]
    #         y = self.track.vertices[0][0]
    #     self.car_state.position_m = Vector2(x, y) # todo reset to start line of track
