"""
Client racecar agent

"""
import argparse
import os

import argcomplete as argcomplete
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import logging
import socket, pickle
import pygame.freetype  # Import the freetype module.

import pickle
import socket
import time
import pygame
from src.globals import *
from src.my_joystick import my_joystick
from src.my_keyboard import my_keyboard
from src.track import track
from src.car import car
from src.my_args import client_args
from src.my_logger import my_logger

logger=my_logger(__name__)

# may only apply to windows
try:
    from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception:
    logger.warning('Gooey GUI builder not available, will use command line arguments.\n'
                   'Install with "pip install Gooey". See README')


def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        epilog='Run with no arguments to open dialog for server IP', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = client_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


class Game:
    def __init__(self, widthPixels=SCREEN_WIDTH_PIXELS, heightPixels=SCREEN_HEIGHT_PIXELS):
        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.ticks = FPS # frame/animation/simulation rate of client (but dt is computed on server real time)
        self.exit = False
        self.input=None

        self.track=track()
        self.car = None # will make it later after we get info from server about car
        try:
            self.input=my_joystick()
        except:
            self.input=my_keyboard()
        # self.track=Track() # TODO for now just use default track # (Marcin) I think this line is redundant here

    def render_multi_line(self, text, x, y): # todo clean up
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.game_font.render_to(self.screen, (x, y + GAME_FONT_SIZE * i), l, [255,255,255]),

    def run(self):
        iterationCounter=0
        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        serverAddr=(SERVER_HOST, SERVER_PORT)
        serverSock.settimeout(SOCKET_TIMEOUT_SEC)
        serverSock.bind(('',0)) # bind to receive on any port from server - seems to cause 'ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host'

        logger.info('connecting to l2race model server at '+str(serverAddr))

        gotServer=False
        while not gotServer :
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True # TODO clean up, seems redundant. what sets pygame.QUIT?
            command = self.input.read()
            if command.quit:
                logger.info('startup aborted before connecting to server')
                pygame.quit()
            cmd='newcar'
            p=pickle.dumps(cmd)
            logger.info('sending cmd={} to server initial address {}, waiting for server'.format(cmd,serverAddr))
            serverSock.sendto(p,serverAddr)
            try:
                data,gameSockAddr=serverSock.recvfrom(4096) # todo add timeout for flaky connection
                gotServer=True
                self.car = car(track=self.track)
                self.car.car_state=pickle.loads(data) # server sends initial state of car
                self.car.loadAndScaleCarImage()
                logger.info('received car server response and initial car state; will use {} for communicating with l2race model server'.format(gameSockAddr))
                logger.info('initial car state is '+str(self.car.car_state))
            except OSError as err:
                s='{}:\n error for response from {}; will try again...'.format(err,serverAddr)
                logger.warning(s)
                self.render_multi_line(s, 10, 10)
                pygame.display.flip()

                time.sleep(1)
                continue


            logger.info('starting main loop')
            while not self.exit:
                iterationCounter+=1
                if iterationCounter%CHECK_FOR_JOYSTICK_INTERVAL==0 and not isinstance(self.input, my_joystick):
                    try:
                        self.input = my_joystick() # check for joystick that might get turned on during play
                    except:
                        pass

                # Event queue
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True

                # User input
                command=self.input.read()
                # logger.info(inp)
                if command.quit:
                    logger.info('quit recieved, ending main loop')
                    self.exit=True

                if command.reset:
                    # car state reset handled on server side, here just put in forward gear
                    self.input.car_input.reverse=False

                # send control to server
                data=command # todo add general command structure to msg
                p=pickle.dumps(data)
                serverSock.sendto(p,gameSockAddr)

                # get new car state
                try:
                    data,_=serverSock.recvfrom(4096) # todo, make blocking with timeout to handle dropped packets
                    (dt,cs)=pickle.loads(data) # todo do something with dt to set animation rate
                    self.car.car_state=cs
                except socket.timeout:
                    # the problem is that if we get a timeout, the next solution will take even longer since the step will be even larger, so we get into spiral
                    logger.warning('Timeout on socket receive from server, using previous car state. check server to make sure it is still running')
                except ConnectionResetError:
                    logger.warning('Connection to {} was reset, will look for server again'.format(gameSockAddr))
                    gotServer=False
                    break
                except TypeError as te:
                    logger.warning(str(te)+": ignoring and waiting for next state")
                    continue

                # Drawing
                # self.screen.fill((10, 10, 10))
                # self.track.get_nearest_waypoint_idx(car_state=self.car.car_state)
                # self.track.get_current_angle_to_road(car_state=self.car.car_state)
                # self.track.get_distance_to_nearest_segment(car_state=self.car.car_state)
                self.track.draw(self.screen)
                # print(self.car.car_state.position)
                self.car.draw(self.screen)
                self.render_multi_line(str(self.car.car_state), 10, 10)
                # self.game_font.render_to(self.screen, (10, 10), str(self.car.car_state), (255, 255, 255))
                pygame.display.flip()
                self.clock.tick(self.ticks) # limit runtime to self.ticks Hz update rate


        logger.info('quitting')
        pygame.quit()
        quit()

if __name__ == '__main__':
    try:
        ga = Gooey(get_args, program_name="l2race client", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey"')
    args = get_args()

    SERVER_HOST = args.host
    SERVER_PORT = args.port
    FPS = args.fps
    JOYSTICK_NUMBER=args.joystick
    game = Game()
    game.run()