# class for Car, holds other important stuff
import copy
import math
import os
from math import radians, cos, sin
from typing import Optional, Tuple

import numpy as np
import pygame
import pygame.freetype
from l2race_utils import my_logger
from track import track

from car_state import car_state
from globals import M_PER_PIXEL, G, CAR_NAME, GAME_FONT_NAME, GAME_FONT_SIZE

logger = my_logger(__name__)

def loadAndScaleCarImage(image_name:str, length_m:float, screen:Optional[pygame.Surface]):
    """ loads image for car and scales it to its actual length.
    Car image should be horizontal and car should be facing to the right.

    Call only after car_state is filled by server since we need physical length and width.
    If passed screen argument, then the image surface is optimized by .convert() to the surface for faster drawing.

    :param image_name: the name of image in the media/cars folder without png suffix
    :param screen: the pygame surface

    :returns: the pygame image that can be rotated and blitted
    """
    if image_name.endswith('.png'):
        logger.info('you can supply car image name {} without .png suffix'.format(image_name))
        image_name=image_name[:-4]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "../media/cars/" + image_name + ".png")
    if isinstance(screen,pygame.Surface):
        image = pygame.image.load(image_path).convert_alpha(screen)  # load image of car
    else:
        image = pygame.image.load(image_path)  # load image of car

    # scale it to its length in pixels (units are in pixels which are scaled from meters by M_PER_PIXEL)
    rect = image.get_rect()
    sc = length_m / (M_PER_PIXEL * rect.width)
    image = pygame.transform.scale(image, (int(sc * rect.width), int(sc * rect.height)))
    return image

class car:
    """
    Local model of car. It has car_state() that is updated by remote server, and methods for drawing car and other static information related to car that is not transmitted over socket.
    """

    def __init__(self, name=CAR_NAME,
                 image_name='car_red',
                 our_track:Optional[track]=None,
                 screen:pygame.surface=None,
                 client_ip:Tuple[str,int]=None):
        ''' Constructs a new car.

        :param name: - the car name
        :param image_name: - the image name, without trailing .png
        :param our_track: - existing track() instance
        :param screen: - the pygame drawing surface
        :param client_ip: - our IP address
        '''
        self.car_state = car_state(name=name, client_ip=client_ip)

        self.track=our_track
        self.image_name = image_name
        self.image=loadAndScaleCarImage(image_name=image_name, length_m=self.car_state.static_info.length_m, screen=screen)
        pygame.freetype.init()
        self.game_font = pygame.freetype.SysFont(name = GAME_FONT_NAME, size = GAME_FONT_SIZE)
        # self.rect = self.image.get_rect()

    def draw(self, screen):
        """ Draws the car
        :param screen: the pygame surface
        """
        if self.image is None:
            logger.warning('no car image yet, cannot draw it')
            return
        # draw our car
        # Find remainder of body angle after dividing by 2pi since large angles result in out of memory error.
        # And model instability results in huge angles sometimes.
        angle_remainder=np.remainder(self.car_state.body_angle_deg,360)
        rotated = pygame.transform.rotate(self.image, -angle_remainder)
        rect = rotated.get_rect()

        p=self.car_state.position_m
        if math.isnan(p.x) or math.isnan(p.y) or math.isnan(self.car_state.body_angle_deg):
            logger.warning('cannot draw ghost car; some value of position or angle is NaN')
            return

        screen.blit(rotated, ((self.car_state.position_m/M_PER_PIXEL) - (int(rect.width / 2), int(rect.height / 2))))
        # label name
        self.game_font.render_to(screen, (self.car_state.position_m.x/M_PER_PIXEL, self.car_state.position_m.y/M_PER_PIXEL), self.car_state.static_info.name, [200,200,200]),

        # draw acceleration
        car_length=(self.car_state.accel_m_per_sec_2.x/G)*(self.car_state.static_info.length_m * 6) # self.car_state.command.throttle*self.car_state.length*2 # todo fix when accel include lateral component
        body_rad=radians(self.car_state.body_angle_deg)
        body_vec=(car_length*cos(body_rad),car_length*sin(body_rad))
        pygame.draw.line(screen, [255,50,50],self.car_state.position_m/M_PER_PIXEL, (self.car_state.position_m+body_vec)/M_PER_PIXEL,1)

        # draw steering command
        str_len= self.car_state.static_info.length_m / 2
        str_orig=self.car_state.position_m+(cos(body_rad)*str_len,sin(body_rad)*str_len)
        str_rad=radians(self.car_state.body_angle_deg+self.car_state.steering_angle_deg)
        str_vec=(str_len*cos(str_rad),str_len*sin(str_rad))
        str_pos1=str_orig-str_vec
        str_pos2=str_orig+str_vec
        pygame.draw.line(screen, [50,250,250],str_pos1/M_PER_PIXEL, str_pos2/M_PER_PIXEL,2)

        # draw speed vector taking into account drift angle
        # speed_angle_rad = radians(self.car_state.body_angle_deg + self.car_state.drift_angle_deg)
        # speed_vec = (self.car_state.speed_m_per_sec*cos(speed_angle_rad), self.car_state.speed_m_per_sec*sin(speed_angle_rad))
        # pygame.draw.line(screen, [255,165,0], self.car_state.position_m/M_PER_PIXEL, (self.car_state.position_m+speed_vec)/M_PER_PIXEL,2)       

        # draw lateral component of speed vector
        drift_angle_rad = radians(self.car_state.drift_angle_deg)
        lateral_speed = self.car_state.speed_m_per_sec * sin(drift_angle_rad)
        lateral_speed_angle_rad = radians(self.car_state.body_angle_deg + 90)
        lateral_speed_vec = (lateral_speed*cos(lateral_speed_angle_rad), lateral_speed*sin(lateral_speed_angle_rad))
        pygame.draw.line(screen, [255,165,0], self.car_state.position_m/M_PER_PIXEL, (self.car_state.position_m + lateral_speed_vec)/M_PER_PIXEL,2)       

        # other cars are drawn by client now using its list of spectate_cars
    #     self.draw_other_cars(screen)
    #
    # def draw_other_car(self, screen:pygame.surface, state:car_state):
    #     if self.other_cars_image is None:
    #         self.other_cars_image=self.loadAndScaleCarImage('other_car',screen)
    #
    #     rotated = pygame.transform.rotate(self.other_cars_image, -state.body_angle_deg)
    #     rect = rotated.get_rect()
    #     screen.blit(rotated, ((state.position_m/M_PER_PIXEL) - (int(rect.width / 2), int(rect.height / 2))))
    #     # label name
    #     self.game_font.render_to(screen, (state.position_m.x/M_PER_PIXEL, state.position_m.y/M_PER_PIXEL), state.static_info.name, [200,200,200]),

    def name(self) -> str:
        """
        Convenience method to get car name from self.car_state

        :return: name of car
        """
        return self.car_state.name()

