# class for Car, holds other important stuff
import os
from math import radians, cos, sin
from typing import Optional, Tuple

import pygame
import pygame.freetype
from pygame.math import Vector2
from src.l2race_utils import my_logger
from src.track import track

from src.car_state import car_state
from src.globals import M_PER_PIXEL, G, CAR_NAME, GAME_FONT_NAME, GAME_FONT_SIZE

logger = my_logger(__name__)


class car:
    """
    Local model of car. It has car_state() that is updated by remote server, and methods for drawing car and other static information related to car that is not transmitted over socket.
    """

    def __init__(self, name=CAR_NAME,
                 image_name='car_1',
                 track:Optional[track]=None,
                 screen:pygame.surface=None,
                 client_ip:Tuple[str,int]=None):
        ''' Constructs a new car.

        :param name - the car name
        :param image_name - the image name, without trailing .png
        :param track - existing track() instance
        :param screen - the pygame drawing surface
        :param client_ip - our IP address
        '''
        self.car_state = car_state(name=name, client_ip=client_ip)

        self.track=track
        self.image_name = image_name
        self.image=self.loadAndScaleCarImage(image_name, screen)
        pygame.freetype.init()
        self.game_font = pygame.freetype.SysFont(name = GAME_FONT_NAME, size = GAME_FONT_SIZE)
        self.other_cars_image=None
        # self.rect = self.image.get_rect()

    def draw(self, screen):
        if self.image is None:
            logger.warning('no car image yet, cannot draw it')
            return
        # draw our car
        rotated = pygame.transform.rotate(self.image, -self.car_state.body_angle_deg)
        rect = rotated.get_rect()
        screen.blit(rotated, ((self.car_state.position_m/M_PER_PIXEL) - (int(rect.width / 2), int(rect.height / 2))))
        # label name
        self.game_font.render_to(screen, (self.car_state.position_m.x/M_PER_PIXEL, self.car_state.position_m.y/M_PER_PIXEL), self.car_state.static_info.name, [200,200,200]),

        # draw acceleration
        len=(self.car_state.accel_m_per_sec_2.x/G)*(self.car_state.static_info.length_m * 6) # self.car_state.command.throttle*self.car_state.length*2 # todo fix when accel include lateral component
        body_rad=radians(self.car_state.body_angle_deg)
        body_vec=(len*cos(body_rad),len*sin(body_rad))
        pygame.draw.line(screen, [255,50,50],self.car_state.position_m/M_PER_PIXEL, (self.car_state.position_m+body_vec)/M_PER_PIXEL,1)

        # draw steering command
        str_len= self.car_state.static_info.length_m / 2
        str_orig=self.car_state.position_m+(cos(body_rad)*str_len,sin(body_rad)*str_len)
        str_rad=radians(self.car_state.body_angle_deg+self.car_state.steering_angle_deg)
        str_vec=(str_len*cos(str_rad),str_len*sin(str_rad))
        str_pos1=str_orig-str_vec
        str_pos2=str_orig+str_vec
        pygame.draw.line(screen, [50,250,250],str_pos1/M_PER_PIXEL, str_pos2/M_PER_PIXEL,2)

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



    def loadAndScaleCarImage(self, image_name:str, screen:Optional[pygame.Surface]):
        """ loads image for car and scales it to its actual length.
        Car image should be horizontal and car should be facing to the right.

        Call only after car_state is filled by server since we need phyical length and width.
        If passed screen argument, then the image surface is optimized by .convert() to the surface for faster drawing.

        """
        if image_name.endswith('.png'):
            logger.info('you can supply car image name {} without .png suffix'.format(image_name))
            image_name=image_name[:-4]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "../media/cars" + image_name + ".png")
        if isinstance(screen,pygame.Surface):
            image = pygame.image.load(image_path).convert_alpha(screen)  # load image of car
        else:
            image = pygame.image.load(image_path)  # load image of car

        # scale it to its length in pixels (units are in pixels which are scaled from meters by M_PER_PIXEL)
        rect = image.get_rect()
        sc = self.car_state.static_info.length_m / (M_PER_PIXEL * rect.width)
        image = pygame.transform.scale(image, (int(sc * rect.width), int(sc * rect.height)))
        return image
