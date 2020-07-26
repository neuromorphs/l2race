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
    def __init__(self, track_name='track', game_mode=1, server_host=SERVER_HOST, server_port=SERVER_PORT, joystick_number=JOYSTICK_NUMBER, fps=FPS, widthPixels=SCREEN_WIDTH_PIXELS, heightPixels=SCREEN_HEIGHT_PIXELS, timeout_s=SERVER_TIMEOUT_SEC):
        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.exit = False
        self.input=None
        self.fps=fps
        self.server_host=server_host
        self.server_port=server_port
        self.server_timeout_s=timeout_s

        self.track_name = track_name
        self.game_mode = game_mode
        self.track = track(track_name=track_name)
        self.car = None # will make it later after we get info from server about car
        try:
            self.input=my_joystick(joystick_number)
        except:
            self.input=my_keyboard()
        # self.track=Track() # TODO for now just use default track # (Marcin) I think this line is redundant here

    def render_multi_line(self, text, x, y): # todo clean up
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.game_font.render_to(self.screen, (x, y + GAME_FONT_SIZE * i), l, [200,200,200]),

    def run(self):
        iterationCounter=0
        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        serverAddr=(self.server_host, self.server_port)
        serverSock.settimeout(self.server_timeout_s)
        serverSock.bind(('',0)) # bind to receive on any port from server - seems to cause 'ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host'

        logger.info('connecting to l2race model server at '+str(serverAddr))

        gotServer=False
        ntries=0
        while not gotServer :
            ntries+=1
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True # TODO clean up, seems redundant. what sets pygame.QUIT?
                    break
            command = self.input.read()
            if command.quit:
                logger.info('startup aborted before connecting to server')
                pygame.quit()
            cmd = 'newcar'
            data = (cmd, self.track_name, self.game_mode)
            p = pickle.dumps(data)
            s='sending cmd={} to server initial address {}, waiting for server...[{}]'.format(cmd, serverAddr,ntries)
            logger.info(s)
            self.screen.fill([0, 0, 0])
            self.render_multi_line(s, 10, 10)
            pygame.display.flip()
            serverSock.sendto(p, serverAddr)
            try:
                p, gameSockAddr = serverSock.recvfrom(4096) # todo add timeout for flaky connection
                (car_state, car_name) = pickle.loads(p)
                gotServer = True
                self.car = car(car_name=car_name)
                self.car.car_state = car_state  # server sends initial state of car
                self.car.loadAndScaleCarImage()
                logger.info('received car server response and initial car state; will use {} for communicating with l2race model server'.format(gameSockAddr))
                logger.info('initial car state is '+str(self.car.car_state))
            except OSError as err:
                s='{}:\n error for response from {}; will try again in {}s ...[{}]'.format(err,serverAddr, SERVER_PING_INTERVAL_S,ntries)
                logger.warning(s)
                self.screen.fill([0,0,0])
                self.render_multi_line(s, 10, 10)
                pygame.display.flip()

                time.sleep(SERVER_PING_INTERVAL_S)
                continue


            logger.info('starting main loop')
            while not self.exit and gotServer:
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
                    break

                if command.reset_car:
                    # car state reset handled on server side, here just put in forward gear
                    self.input.car_input.reverse=False

                if command.restart_client:
                    logger.info('restarting client')
                    gotServer=False
                    break

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
                self.track.draw(self.screen)
                self.car.draw(self.screen)
                self.render_multi_line(str(self.car.car_state), 10, 10)
                pygame.display.flip()
                self.clock.tick(self.fps) # limit runtime to self.ticks Hz update rate

        if serverSock:
            logger.info('closing socket')
            serverSock.close()
        logger.info('quitting pygame')
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

    game = Game(track_name='track', game_mode=1, server_host=args.host, server_port=args.port, joystick_number=args.joystick, fps=args.fps, timeout_s=args.timeout_s)
    game.run()