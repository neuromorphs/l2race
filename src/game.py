import os
import pickle
import socket
from math import sin, radians, degrees, copysign

import pygame
from pygame.math import Vector2

import logging

from src.network import getudpsocket, HOST
from src.joystick import Joystick
from src.keyboard import Keyboard

logger = logging.getLogger(__name__)

WIDTH=1400
HEIGHT=1000
PPU=32

class Game:
    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        pygame.display.set_caption("l2race")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.input=None
        try:
            self.input=Joystick()
        except:
            self.input=Keyboard()

    def run(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "../media/car.png")
        car_image = pygame.image.load(image_path)
        sc=.5
        rect = car_image.get_rect()
        car_image=pygame.transform.scale(car_image,(int(sc*rect.width),int(sc*rect.height)))

        from src.car import Car
        car = Car(0, 0)
        ppu = PPU

        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        serverAddr=('localhost', 50000)

        cmd='newcar'
        p=pickle.dumps(cmd)
        logger.info('sending cmd={}'.format(cmd))
        serverSock.sendto(p,serverAddr)
        p,gameSockAddr=serverSock.recvfrom(4096)
        # gameSockAddr=pickle.loads(p)
        logger.info('received car server address={}'.format(gameSockAddr))

        while not self.exit:
            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            inp=self.input.read()

            c=(dt,inp)
            p=pickle.dumps(c)
            serverSock.sendto(p,gameSockAddr)

            r=serverSock.recv(4096)
            cs=pickle.loads(r)
            car.car_state=cs

            # Drawing
            self.screen.fill((10, 10, 10))
            rotated = pygame.transform.rotate(car_image, car.car_state.angle_deg)
            rect = rotated.get_rect()
            self.screen.blit(rotated, ((car.car_state.position * ppu) - (int(rect.width / 2), int(rect.height / 2))))
            pygame.display.flip()

            self.clock.tick(self.ticks)

        pygame.quit()


