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
import sys

from src.data_recorder import data_recorder
from src.l2race_utils import bind_socket_to_range, open_ports
from src.globals import *
from src.my_joystick import my_joystick
from src.my_keyboard import my_keyboard
from src.track import track
from src.car import car
from src.my_args import client_args
from src.my_logger import my_logger
from src.pid_next_waypoint_car_controller import pid_next_waypoint_car_controller

logger = my_logger(__name__)





def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        epilog='Run with no arguments to open dialog for server IP', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = client_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def launch_gui():
    # may only apply to windows
    try:
        from scripts.regsetup import description
        from gooey import Gooey  # pip install Gooey
    except Exception:
        logger.warning('Gooey GUI builder not available, will use command line arguments.\n'
                       'Install with "pip install Gooey". See README')

    try:
        ga = Gooey(get_args, program_name="l2race client", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey".\n'
                    'Ignore this warning if you do not want a GUI.')


class Game:

    def __init__(self,
                 track_name='track',
                 game_mode=GAME_MODE,
                 car_name=CAR_NAME,
                 controller=pid_next_waypoint_car_controller(),
                 server_host=SERVER_HOST,
                 server_port=SERVER_PORT,
                 joystick_number=JOYSTICK_NUMBER,
                 fps=FPS,
                 widthPixels=SCREEN_WIDTH_PIXELS,
                 heightPixels=SCREEN_HEIGHT_PIXELS,
                 timeout_s=SERVER_TIMEOUT_SEC,
                 record=DATA_FILENAME_BASE):
        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        pygame.freetype.init()
        self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.exit = False
        self.input=None
        self.fps=fps
        self.server_host=server_host
        self.server_port=server_port
        self.server_timeout_s=timeout_s
        self.record=record
        self.recorder=None

        self.track_name = track_name
        self.game_mode = game_mode
        self.track = track(track_name=track_name)
        self.car_name=car_name
        self.car = None # will make it later after we get info from server about car
        try:
            self.input = my_joystick(joystick_number)
        except:
            self.input = my_keyboard()

        # if controller is None:
        #     self.controller = pid_next_waypoint_car_controller()
        # else:
        #     self.controller = controller

        self.controller = controller

        self.auto_input = None  # will make it later when car is created because it is needed for the car_controller

        self.serverSock = None
        self.gotServer = None
        self.gameSockAddr = None


    def render_multi_line(self, text, x, y): # todo clean up
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.game_font.render_to(self.screen, (x, y + GAME_FONT_SIZE * i), l, [200,200,200]),
            pass

    def connect_to_server(self):
        try:
            open_ports()
        except Exception as ex:
            logger.warning("Caught exception {} when trying to open l2race client ports".format(ex))

        self.serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        serverAddr = (self.server_host, self.server_port)
        self.serverSock.settimeout(self.server_timeout_s)
        bind_socket_to_range(CLIENT_PORT_RANGE, self.serverSock)

        logger.info('connecting to l2race model server at '+str(serverAddr))

        self.gotServer=False
        ntries = 0
        while not self.gotServer:
            ntries += 1
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
            payload=(self.track_name, self.game_mode, self.car_name)
            data = (cmd, payload)
            p = pickle.dumps(data)
            s = 'sending cmd={} with payload {} to server initial address {}, ' \
                'waiting for server...[{}]'\
                .format(cmd, payload, serverAddr, ntries)
            logger.info(s)
            self.screen.fill([0, 0, 0])
            self.render_multi_line(s, 10, 10)
            pygame.display.flip()
            self.serverSock.sendto(p, serverAddr)
            try:
                p, self.gameSockAddr = self.serverSock.recvfrom(4096) # todo add timeout for flaky connection
                car_state = pickle.loads(p)
                self.gotServer = True
                self.car = car(name=self.car_name)
                self.car.car_state = car_state  # server sends initial state of car
                self.car.track = track(track_name=self.track_name)
                if self.record:
                    if self.recorder is None:
                        self.recorder = data_recorder(car=self.car)
                    self.recorder.open_new_recording()
                self.car.loadAndScaleCarImage()
                self.controller.car=self.car
                self.auto_input =self.controller
                logger.info('received car server response and initial car state; '
                            'will use {} for communicating with l2race model server'.format(self.gameSockAddr))
                logger.info('initial car state is '+str(self.car.car_state))
            except OSError as err:
                s = '{}:\n error for response from {}; ' \
                    'will try again in {}s ...[{}]'.format(err, serverAddr, SERVER_PING_INTERVAL_S, ntries)
                logger.warning(s)
                self.screen.fill([0, 0, 0])
                self.render_multi_line(s, 10, 10)
                pygame.display.flip()

                time.sleep(SERVER_PING_INTERVAL_S)
                continue

    def run(self):

        self.connect_to_server()

        iterationCounter = 0
        logger.info('starting main loop')
        while not self.exit and self.gotServer:
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
            external_input = self.input.read()
            if external_input.auto:
                command = self.auto_input.read()
                command.throttle = external_input.throttle
                command.reset_car = external_input.reset_car
                command.restart_client = external_input.restart_client
                command.quit = external_input.quit
                command.auto = external_input.auto
            else:
                command = external_input

            if command.quit:
                logger.info('quit recieved, ending main loop')
                self.exit=True
                break

            if command.reset_car:
                # car state reset handled on server side, here just put in forward gear
                logger.info('sending message to reset car state to server, putting in foward gear on client')
                self.input.car_input.reverse=False

            if command.restart_client:
                logger.info('restarting client')
                self.gotServer = False
                self.connect_to_server()
                if self.recorder:
                    self.recorder.close_recording()
                break

            # send control to server
            data = command # todo add general command structure to msg
            p = pickle.dumps(data)
            self.serverSock.sendto(p,self.gameSockAddr)

            # get new car state
            try:
                data, _ = self.serverSock.recvfrom(4096) # todo, make blocking with timeout to handle dropped packets
                (dt, cs) = pickle.loads(data) # todo do something with dt to set animation rate
                self.car.car_state = cs
            except socket.timeout:
                # the problem is that if we get a timeout,
                # the next solution will take even longer since the step will be even larger, so we get into spiral
                logger.warning('Timeout on socket receive from server, using previous car state. '
                               'Check server to make sure it is still running')
            except ConnectionResetError:
                logger.warning('Connection to {} was reset, will look for server again'.format(self.gameSockAddr))
                self.gotServer = False
                self.connect_to_server()
                break
            except TypeError as te:
                logger.warning(str(te)+": ignoring and waiting for next state")
                continue

            if self.recorder:
                self.recorder.write_sample()
            # Drawing
            self.car.track.draw(self.screen)
            self.car.draw(self.screen)
            self.render_multi_line(str(self.car.car_state), 10, 10)
            pygame.display.flip()
            self.clock.tick(self.fps) # limit runtime to self.ticks Hz update rate

        if self.serverSock:
            logger.info('closing socket')
            self.serverSock.close_recording()
        logger.info('quitting pygame')
        pygame.quit()
        quit()


# A wrapper around Game class to make it easier for a user to provide arguments
def define_game(gui='with_gui',
                track_name=None,
                game_mode=None,
                car_name=None,
                server_host=None,
                server_port=None,
                joystick_number=None,
                fps=None,
                timeout_s=None,
                record=None):

    if gui == 'with_gui':
        launch_gui()
        args = get_args()
        game = Game(track_name=args.track_name,
                    game_mode='multi' if args.multi else 'solo',
                    car_name=args.car_name,
                    # server_host=args.host,
                    # server_port=args.port,
                    joystick_number=args.joystick,
                    # fps=args.fps,
                    # timeout_s=args.timeout_s,
                    record=args.record)
    else:

        IGNORE_COMMAND = '--ignore-gooey'
        if IGNORE_COMMAND in sys.argv:
            sys.argv.remove(IGNORE_COMMAND)

        args = get_args()

        if track_name is None:
            track_name = args.track_name

        if game_mode is None:
            game_mode = 'multi' if args.multi else 'solo'

        if car_name is None:
            car_name = args.car_name

        try:
            if server_host is None:
                server_host = args.host
        except NameError:
            server_host = args.host

        try:
            if server_port is None:
                server_port = args.port
        except NameError:
            server_port = args.port

        if joystick_number is None:
            joystick_number = args.joystick

        try:
            if fps is None:
                fps = args.fps
        except NameError:
            fps = args.fps

        try:
            if timeout_s is None:
                timeout_s = args.timeout_s
        except NameError:
            timeout_s = args.timeout_s

        if record is None:
            record = args.record

        game = Game(track_name=track_name,
                    game_mode=game_mode,
                    car_name=car_name,
                    server_host=server_host,
                    server_port=server_port,
                    joystick_number=joystick_number,
                    fps=fps,
                    timeout_s=timeout_s,
                    record=record)

    return game




if __name__ == '__main__':

    launch_gui()

    args = get_args()

    game = Game(track_name=args.track_name,
                game_mode='multi' if args.multi else 'solo',
                car_name=args.car_name,
                server_host=args.host,
                server_port=args.port,
                joystick_number=args.joystick,
                fps=args.fps,
                timeout_s=args.timeout_s,
                record=args.record)

    # game = define_game()

    game.run()