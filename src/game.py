import os
import pickle
import socket
import time
from math import sin, radians, degrees, copysign

import pygame
from pygame.math import Vector2

import logging

from src.globals import SCREEN_WIDTH, SCREEN_HEIGHT, PPU
from src.network import getudpsocket, HOST
from src.joystick import Joystick
from src.keyboard import Keyboard
from src.track import Track

logger = logging.getLogger(__name__)

CHECK_FOR_JOYSTICK_INTERVAL = 100


class Game: # todo move to client
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(size=(self.width, self.height),flags=0)
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.input=None
        try:
            self.input=Joystick()
        except:
            self.input=Keyboard()
        self.track=Track() # TODO for now just use default track

    def run(self):

        iterationCounter=0
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "../media/car.png")
        car_image = pygame.image.load(image_path)
        sc=.5
        rect = car_image.get_rect()
        car_image=pygame.transform.scale(car_image,(int(sc*rect.width),int(sc*rect.height)))

        from src.car import Car
        car = Car()

        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        serverAddr=('localhost', 50000)

        gotServer=False
        while not gotServer:
            cmd='newcar'
            p=pickle.dumps(cmd)
            logger.info('sending cmd={}, waiting for server'.format(cmd))
            serverSock.sendto(p,serverAddr)
            try:
                p,gameSockAddr=serverSock.recvfrom(4096) # todo add timeout for flaky connection
                gotServer=True
                logger.info('received car server address={}'.format(gameSockAddr))
            except:
                logger.warning('no response; waiting...')
                time.sleep(1)


        while not self.exit:
            iterationCounter+=1
            if iterationCounter%CHECK_FOR_JOYSTICK_INTERVAL==0 and not isinstance(self.input,Joystick):
                try:
                    self.input = Joystick() # check for joystick that might get turned on during play
                except:
                    pass

            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            inp=self.input.read()
            # logger.info(inp)

            c=(dt,inp) # todo add general command structure to msg
            p=pickle.dumps(c)
            serverSock.sendto(p,gameSockAddr)

            r=serverSock.recv(4096) # todo, make blocking with timeout to handle dropped packets
            cs=pickle.loads(r)
            car.car_state=cs

            # Drawing
            self.screen.fill((10, 10, 10))
            car.track.draw(self.screen)
            rotated = pygame.transform.rotate(car_image, car.car_state.angle_deg)
            rect = rotated.get_rect()
            self.screen.blit(rotated, ((car.car_state.position ) - (int(rect.width / 2), int(rect.height / 2))))
            pygame.display.flip()

            self.clock.tick(self.ticks)

            if inp.quit:
                logger.info('quit recieved, ending main loop')
                self.exit=True


        logger.info('quitting')
        pygame.quit()
        quit()


