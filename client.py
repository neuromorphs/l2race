"""
Client racecar agent

"""
import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import logging
import socket, pickle

from car_state import CarState
from src.mylogger import mylogger

logger=mylogger(__name__)
import pickle
import socket
import time
import pygame
from src.globals import *
from src.joystick import Joystick
from src.keyboard import Keyboard
from src.track import Track
from src.car import Car

CHECK_FOR_JOYSTICK_INTERVAL = 100


class Game:
    def __init__(self, widthPixels=SCREEN_WIDTH_PIXELS, heightPixels=SCREEN_HEIGHT_PIXELS):
        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        self.clock = pygame.time.Clock()
        self.ticks = 20 # frame/animation/simulation rate of client (but dt is computed on server real time)
        self.exit = False
        self.input=None

        self.track=Track()
        self.car = None # will make it later after we get info from server about car
        try:
            self.input=Joystick()
        except:
            self.input=Keyboard()
        # self.track=Track() # TODO for now just use default track # (Marcin) I think this line is redundant here

    def run(self):
        iterationCounter=0
        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        serverAddr=(SERVER_HOST, SERVER_PORT)
        serverSock.settimeout(SOCKET_TIMEOUT_SEC)

        logger.info('connecting to server at '+str(serverAddr))

        gotServer=False
        while not gotServer :
            inp = self.input.read()
            if inp.quit:
                logger.info('startup aborted before connecting to server')
                pygame.quit()
            cmd='newcar'
            p=pickle.dumps(cmd)
            logger.info('sending cmd={} to {}, waiting for server'.format(cmd,serverAddr))
            serverSock.sendto(p,serverAddr)
            try:
                data,gameSockAddr=serverSock.recvfrom(4096) # todo add timeout for flaky connection
                gotServer=True
                self.car = Car(track=self.track)
                self.car.car_state=pickle.loads(data) # server sends initial state of car
                self.car.loadAndScaleCarImage()
                logger.info('received car server address={}'.format(gameSockAddr))
            except:
                logger.warning('no response; waiting...')
                time.sleep(1)
                continue


            logger.info('starting main loop')
            while not self.exit:
                iterationCounter+=1
                if iterationCounter%CHECK_FOR_JOYSTICK_INTERVAL==0 and not isinstance(self.input,Joystick):
                    try:
                        self.input = Joystick() # check for joystick that might get turned on during play
                    except:
                        pass

                dt = self.clock.get_time() / 1000 # todo move dt to server, which is in charge of dynamics

                # Event queue
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True

                # User input
                inp=self.input.read()
                # logger.info(inp)
                if inp.quit:
                    logger.info('quit recieved, ending main loop')
                    self.exit=True

                # send control to server
                data=inp # todo add general command structure to msg
                p=pickle.dumps(data)
                serverSock.sendto(p,gameSockAddr)

                # get new car state
                try:
                    data,_=serverSock.recvfrom(4096) # todo, make blocking with timeout to handle dropped packets
                    (dt,cs)=pickle.loads(data) # todo do something with dt to set animation rate
                    self.car.car_state=cs
                except socket.timeout:
                    logger.warning('Timeout on socket receive from server, using previous car state. check server to make sure it is still running')
                except ConnectionResetError:
                    logger.warning('Connection was reset, will look for server again')
                    gotServer=False
                    break

                # Drawing
                self.screen.fill((10, 10, 10))
                self.track.draw(self.screen)
                # print(self.car.car_state.position)
                self.car.draw(self.screen)
                pygame.display.flip()

                self.clock.tick(self.ticks) # limit runtime to self.ticks Hz update rate


        logger.info('quitting')
        pygame.quit()
        quit()

if __name__ == '__main__':
    game = Game()
    game.run()