"""
Client l2race agent

"""
import argparse
import os
from typing import Tuple


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
                 spectate=False,
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
        self.spectate=spectate
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        pygame.freetype.init()
        self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.exit = False
        self.input=None
        self.fps=fps
        self.sock = None # our socket used for communicating with server
        self.server_host=server_host
        self.server_port=server_port
        self.serverStartAddr = (self.server_host, self.server_port) # manager address, different port on server used during game
        self.gameSockAddr = None # address used during game
        self.server_timeout_s=timeout_s
        self.gotServer = None
        self.record=record
        self.recorder=None

        self.track_name = track_name
        self.game_mode = spectate
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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.sock.settimeout(self.server_timeout_s)
        bind_socket_to_range(CLIENT_PORT_RANGE, self.sock)

        logger.info('asking l2race model server at '+str(self.serverStartAddr)+' to add car or spectate')

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
            if not self.spectate:
                cmd = 'add_car'
                payload=(self.track_name, self.car_name)
            else:
                cmd = 'add_spectator'
                payload=self.track_name
            s = 'sending cmd={} with payload {} to server initial address {}, ' \
                'waiting for server...[{}]' \
                .format(cmd, payload, self.serverStartAddr, ntries)
            logger.info(s)
            self.send_to_server(self.serverStartAddr, cmd,payload)

            # now get the game port as a response


            self.screen.fill([0, 0, 0])
            self.render_multi_line(s, 10, 10)
            pygame.display.flip()
            try:
                logger.debug('waiting for game_port message from server')
                msg,payload=self.receive_from_server()
                if msg!='game_port' or not isinstance(payload,int):
                    logger.warning("got response (msg,command)=({},{}) but expected ('game_port',port_number); will try again in {}s".format(msg,payload, SERVER_PING_INTERVAL_S))
                    time.sleep(SERVER_PING_INTERVAL_S)
                    continue
                self.gotServer = True
                self.gameSockAddr=(self.server_host, payload)
                logger.info('got game_port message from server telling us to use address {} to talk with server'.format(self.gameSockAddr))
                self.car = car(name=self.car_name, screen=self.screen)
                self.car.track = track(track_name=self.track_name)
                if self.record:
                    if self.recorder is None:
                        self.recorder = data_recorder(car=self.car)
                    self.recorder.open_new_recording()
                # self.car.loadAndScaleCarImage()   # happens inside car
                self.controller.car=self.car
                self.auto_input =self.controller
                logger.info('initial car state is '+str(self.car.car_state))
                logger.info('received car server response and initial car state; '
                            'will use {} for communicating with l2race model server'.format(self.gameSockAddr))
            except OSError as err:
                s += '\n{}:\n error for response from {}; ' \
                    'will try again in {}s ...[{}]'.format(err, self.serverStartAddr, SERVER_PING_INTERVAL_S, ntries)
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
            self.clock.tick(self.fps) # updates pygame clock, makes sure frame rate is at most this many fps; limit runtime to self.ticks Hz update rate
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
            self.send_to_server(self.gameSockAddr, 'command',command)

             # get new car state
            try:
                cmd,payload=self.receive_from_server()
                if cmd=='car_state':
                    self.car.car_state=payload
                    if self.recorder:
                        self.recorder.write_sample()
                else:
                    logger.warning('unexpected msg {} with payload {} received from server (should have gotten "car_state" message)'.format(cmd,payload))
                    continue
            except pickle.UnpicklingError as err:
                logger.warning('{}: could not unpickle the response from server'.format(err))
                continue
            except TypeError as te:
                logger.warning(str(te)+": ignoring and waiting for next state")
                continue
            except socket.timeout:
                logger.warning('Timeout on socket receive from server, using previous car state. '
                               'Check server to make sure it is still running')
            except ConnectionResetError:
                logger.warning('Connection to {} was reset, will look for server again'.format(self.gameSockAddr))
                self.gotServer = False
                self.connect_to_server()
                break

            # Drawing
            self.draw()

        if self.sock:
            logger.info('closing socket')
            self.sock.close()
        logger.info('quitting pygame')
        pygame.quit()
        quit()

    def send_to_server(self, addr:Tuple[str,int], msg:str, payload:object):
        ''' send cmd,payload to server at specified (ip,port)'''
        logger.debug('sending msg {} with payload {} to {}'.format(msg,payload,addr))
        p = pickle.dumps((msg, payload))
        self.sock.sendto(p, addr)

    def receive_from_server(self):
        data, server_addr = self.sock.recvfrom(8000) # todo check if large enough for payload inclding all other car state
        (cmd,payload) = pickle.loads(data)
        return cmd,payload

    def draw(self):
        self.car.track.draw(self.screen)
        self.car.draw(self.screen)
        self.render_multi_line(str(self.car.car_state), 10, 10)
        pygame.display.flip()


# A wrapper around Game class to make it easier for a user to provide arguments
def define_game(gui='with_gui',
                track_name=None,
                spectate=False,
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
                    spectate=spectate,
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
                    spectate=spectate,
                    car_name=car_name,
                    server_host=server_host,
                    server_port=server_port,
                    joystick_number=joystick_number,
                    fps=fps,
                    timeout_s=timeout_s,
                    record=record)

    return game




if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)
    launch_gui()

    args = get_args()

    game = Game(track_name=args.track_name,
                spectate=args.spectate,
                car_name=args.car_name,
                server_host=args.host,
                server_port=args.port,
                joystick_number=args.joystick,
                fps=args.fps,
                timeout_s=args.timeout_s,
                record=args.record)

    # game = define_game()

    game.run()