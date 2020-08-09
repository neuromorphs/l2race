"""
Client l2race agent

"""
import argparse
import os
from typing import Tuple, List, Optional, Dict

import argcomplete as argcomplete
import pygame
from math import sin, radians, degrees, copysign

import select
from pygame.math import Vector2
import logging
import socket, pickle
import pygame.freetype  # Import the freetype module.

import pickle
import socket
import time
import pygame
import sys
import atexit

from src.car_state import car_state
from src.data_recorder import data_recorder
from src.l2race_utils import find_unbound_port_in_range, open_ports, set_logging_level, loop_timer
from src.globals import *
from src.my_joystick import my_joystick
from src.my_keyboard import my_keyboard
from src.track import track
from src.car import car
from src.my_args import client_args, write_args_info
from src.l2race_utils import my_logger
from src.pid_next_waypoint_car_controller import pid_next_waypoint_car_controller
from src.car_command import car_command

import numpy as np
import pandas as pd
import re
from time import sleep

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


class client:

    def __init__(self,
                 track_name='track',
                 spectate=False,
                 car_name=CAR_NAME,
                 controller=None,
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
        self.sock:Optional[socket] = None # our socket used for communicating with server
        self.server_host:str=server_host
        self.server_port:int=server_port
        self.serverStartAddr: Tuple[str,int] = (self.server_host, self.server_port) # manager address, different port on server used during game
        self.gameSockAddr:Optional[Tuple[str,int]] = None # address used during game
        self.server_timeout_s=timeout_s
        self.gotServer = False
        self.record:bool=record
        self.recorder:Optional[data_recorder]=None

        self.track_name:str = track_name
        self.car_name:str=car_name
        self.car:Optional[car] = None # will make it later after we get info from server about car
        try:
            self.input = my_joystick(joystick_number)
        except:
            self.input = my_keyboard()

        # if controller is None:
        #     self.controller = pid_next_waypoint_car_controller()
        # else:
        #     self.controller = controller

        # spectator data structures
        self.spectate_track:Optional[track] = None # used here for track when there is no car, otherwise track is part of the car object.
        self.spectate_cars:Dict[str,car]=dict() # dict of other cars on the track, by name of the car. Each entry is a car() that we make here.
        self.autodrive_controller = controller  # automatic self driving controller specified in constructor

    def cleanup(self):
        if self.gotServer:
            if self.spectate:
                self.send_to_server(self.gameSockAddr,'remove_spectator',None)
            else:
                self.send_to_server(self.gameSockAddr,'remove_car',self.car_name)
        if self.sock:
            self.sock.close()
            self.sock=None

    def render_multi_line(self, text, x, y): # todo clean up
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.game_font.render_to(self.screen, (x, y + GAME_FONT_SIZE * i), l, [200,200,200]),
            pass

    def ping_server(self):
        logger.info('pinging server at {}'.format(self.serverStartAddr))
        self.drain_udp_messages()
        self.send_to_server(self.serverStartAddr,'ping',None)
        try:
            (msg,payload)=self.receive_from_server()
        except socket.timeout:
            logger.warning('ping timeout')
            return False
        except ConnectionResetError:
            logger.warning('ping connection reset error')
            return False
        except Exception as e:
            logger.warning('ping other exception {}'.format(e))
            return False
        if msg!='pong':
            logger.warning('wrong response {} received for ping'.format(msg))
            return False
        else:
            logger.info('pong received')
            return True

    def connect_to_server(self):
        if self.gotServer:
            return
        logger.info('connecting to l2race model server at '+str(self.serverStartAddr)+' to add car or spectate')
        ntries = 0
        looper=loop_timer(rate_hz=1./SERVER_PING_INTERVAL_S)
        err_str=''
        while not self.gotServer:
            looper.sleep_leftover_time()
            self.screen.fill([0, 0, 0])
            self.render_multi_line(err_str, 10, 10)
            pygame.display.flip()
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
            if not self.ping_server():
                err_str='No response to ping server at {}, will try again in {:.1f}s [{}]'.\
                    format(self.serverStartAddr, 1./looper.rate_hz, looper.loop_counter)
                continue
            if not self.spectate:
                cmd = 'add_car'
                payload=(self.track_name, self.car_name)
            else:
                cmd = 'add_spectator'
                payload=self.track_name
            err_str = 'sending cmd={} with payload {} to server initial address {}, ' \
                'waiting for server...[{}]' \
                .format(cmd, payload, self.serverStartAddr, ntries)
            logger.info(err_str)
            self.screen.fill([0, 0, 0])
            self.render_multi_line(err_str, 10, 10)
            pygame.display.flip()
            self.send_to_server(self.serverStartAddr, cmd,payload)

            try:
                # now get the game port as a response
                logger.info('pausing for server track process to start (if not running already)')
                time.sleep(2) # it takes significant time to start the track process. To avoid timeout, wait a bit here before checking
                logger.info('receiving game_port message from server')
                msg,payload=self.receive_from_server()
                if msg!='game_port':
                    logger.warning("got response (msg,command)=({},{}) but expected ('game_port',port_number); will try again in {}s".format(msg,payload, SERVER_PING_INTERVAL_S))
                    continue
            except OSError as err:
                err_str += '\n{}:\n error for response from {}; ' \
                     'will try again in {}s ...[{}]'.format(err, self.serverStartAddr, SERVER_PING_INTERVAL_S, ntries)
                logger.warning(err_str)
                continue

            self.gotServer = True
            self.gameSockAddr=(self.server_host, payload)
            logger.info('got game_port message from server telling us to use address {} to talk with server'.format(self.gameSockAddr))
            if not self.spectate:
                self.car = car(name=self.car_name, screen=self.screen, client_ip=self.gameSockAddr)
                self.car.track = track(track_name=self.track_name)
                if self.record:
                    if self.recorder is None:
                        self.recorder = data_recorder(car=self.car)
                    self.recorder.open_new_recording()
                # self.car.loadAndScaleCarImage()   # happens inside car
                self.autodrive_controller.car =self.car
                logger.info('initial car state is {}'.format(self.car.car_state))
            else:
                self.spectate_track=track(track_name=self.track_name)

    def run(self):

        try:
            open_ports()
        except Exception as ex:
            logger.warning("Caught exception {} when trying to open l2race client ports".format(ex))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.sock.settimeout(self.server_timeout_s)
        find_unbound_port_in_range(CLIENT_PORT_RANGE)

        atexit.register(self.cleanup)

        iterationCounter = 0
        logger.info('starting main loop')
        looper=loop_timer(self.fps)
        while not self.exit:
            self.connect_to_server() # TODO wrong, will not look again if server connection lost, should go back to this
            try:
                self.clock.tick(self.fps) # updates pygame clock, makes sure frame rate is at most this many fps; limit runtime to self.ticks Hz update rate
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt, stopping client')
                self.exit=True
                continue
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
            if external_input.autodrive_enabled:
                if self.autodrive_controller is None:
                    logger.error('Tried to use autodrive control but there is no controller defined; disabling autodrive')
                    external_input.autodrive_enabled=False
                command = self.autodrive_controller.read()
                command.add_command(external_input)
                command.complete_default()
            else:
                command = external_input
                command.complete_default()

            if command.quit:
                logger.info('quit recieved, ending main loop')
                self.exit=True
                break

            if not self.spectate:
                if command.reset_car:
                    # car state reset handled on server side, here just put in forward gear
                    logger.info('sending message to reset car state to server, putting in foward gear on client')
                    self.input.car_input.reverse=False

                if command.restart_client:
                    logger.info('restarting client')
                    self.gotServer = False
                    if self.recorder:
                        self.recorder.close_recording()
                    continue

                # send control to server
                self.send_to_server(self.gameSockAddr, 'command',command)
            else:
                self.send_to_server(self.gameSockAddr,'send_states',None)

             # expect to get new car state
            try:
                cmd,payload=self.receive_from_server()
                self.handle_message(cmd, payload)
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
                continue

            if not self.spectate and self.recorder:
                self.recorder.write_sample()

            # Drawing
            self.draw()
            try:
                looper.sleep_leftover_time()
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt, stopping client')
                self.exit=True

        logger.info('ending main loop')
        self.cleanup()
        logger.info('quitting pygame')
        pygame.quit()
        quit()

    def send_to_server(self, addr:Tuple[str,int], msg:str, payload:object):
        ''' send cmd,payload to server at specified (ip,port)'''
        if self.sock is None:
            logger.warning('no socket to send message {} with payload {}'.format(msg,payload))
            return
        logger.debug('sending msg {} with payload {} to {}'.format(msg,payload,addr))
        p = pickle.dumps((msg, payload))
        self.sock.sendto(p, addr)

    def drain_udp_messages(self):
        """remove the data present on the socket. From https://stackoverflow.com/questions/1097974/how-to-empty-a-socket-in-python """
        logger.debug('draining existing received UDP messages')
        input = [self.sock]
        while True:
            inputready, o, e = select.select(input,[],[], 0.0)
            if len(inputready)==0: break
            for s in inputready: s.recv(1)

    def receive_from_server(self):
        data, server_addr = self.sock.recvfrom(8192) # todo check if large enough for payload inclding all other car state
        (cmd,payload) = pickle.loads(data)
        logger.debug('got message {} with payload {} from server {}'.format(cmd,payload,server_addr))
        return cmd,payload

    def draw(self):
        if not self.spectate:
            self.draw_car_view()
        else:
            self.draw_spectate_view()
        pygame.display.flip()

    def draw_car_view(self):
        self.car.track.draw(self.screen)
        self.car.draw(self.screen)
        self.render_multi_line(str(self.car.car_state), 10, 10)

    def draw_spectate_view(self):
        self.spectate_track.draw(self.screen)
        for c in self.spectate_cars.values():
            c.draw(self.screen)

    def update_state(self, all_states:List[car_state]):
        to_remove=[]
        for s in all_states:
            pass # todo remove cars that have disappeared from the list of other car states.
        for s in all_states:
            name=s.static_info.name # get the car name from the remote state
            if name==self.car_name:
                self.car.car_state=s # update our own state
            # update other cars on the track
            c=self.spectate_cars.get(name) # get the car
            if c is None: # if it doesn't exist, construct it
                self.spectate_cars[name]=car(name=name,image_name='other_car.png', client_ip=s.static_info.client_ip)
            self.spectate_cars[name].car_state=s # set its state

    def handle_message(self, msg, payload):
        if msg=='state':
            self.update_state(payload)
        elif msg=='game_port':
            self.gameSockAddr=(self.server_host,payload)
        elif msg== 'track_shutdown':
            logger.warning('{}, will try to look for it again'.format(payload))
            self.gotServer=False
        elif msg=='string_message':
            logger.info('recieved message "{}"'.format(payload))
        else:
            logger.warning('unexpected msg {} with payload {} received from server (should have gotten "car_state" message)'.format(msg, payload))

    def replay(self, race_name = None):
        # Load data

        # Find the right file
        if race_name:
            try:
                file_path = './data/'+race_name+'.csv'

            except FileNotFoundError:
                logger.warning('There is no race recording with name {}'.format(race_name))
                return
        else:
            try:
                import glob
                import os
                list_of_files = glob.glob('./data/*.csv')
                file_path = max(list_of_files, key=os.path.getctime)
            except FileNotFoundError:
                logger.warning('No race recording found in data folder')
                return

        # Get race recording
        print(file_path)
        data = pd.read_csv(file_path, skiprows=7)

        # Get used car name
        s = str(pd.read_csv(file_path, skiprows=5, nrows=1))
        self.car_name = re.search('"(.*)"', s).group(1)
        print(self.car_name)

        # Get used track
        s = str(pd.read_csv(file_path, skiprows=6, nrows=1))
        self.track_name = re.search('"(.*)"', s).group(1)
        print(self.track_name)

        # print(file_path)
        # print(data.head())

        # Define car and track
        self.car = car(name=self.car_name, screen=self.screen)
        self.car.track = track(track_name=self.track_name)

        # decimate data
        data = data.iloc[::4, :]

        # Run a loop to print data
        for index, row in data.iterrows():

            self.car.car_state.command.autodrive_enabled = row['cmd.auto']
            self.car.car_state.command.steering = row['cmd.steering']
            self.car.car_state.command.throttle = row['cmd.throttle']
            self.car.car_state.command.brake = row['cmd.brake']
            self.car.car_state.command.reverse = row['cmd.reverse']

            self.car.car_state.time = row['time']
            self.car.car_state.position_m = Vector2(row['pos.x'], row['pos.y'])
            self.car.car_state.velocity_m_per_sec = Vector2(row['vel.x'],row['vel.y'])
            self.car.car_state.speed_m_per_sec = row['speed']
            self.car.car_state.accel_m_per_sec_2 = Vector2(row['accel.x'],row['accel.y'])
            self.car.car_state.steering_angle_deg = row['steering_angle']
            self.car.car_state.body_angle_deg = row['body_angle']
            self.car.car_state.yaw_rate_deg_per_sec = row['yaw_rate']
            self.car.car_state.drift_angle_deg = row['drift_angle']


            # Drawing
            self.draw()
            sleep(0.1)
            # try:
            #     looper.sleep_leftover_time()
            # except KeyboardInterrupt:
            #     logger.info('KeyboardInterrupt, stopping client')
            #     self.exit=True


# A wrapper around Game class to make it easier for a user to provide arguments
def define_game(gui=True,  # set to False to prevent gooey dialog
                track_name=None,
                ctrl=None,
                spectate=False,
                car_name=None,
                server_host=None,
                server_port=None,
                joystick_number=None,
                fps=None,
                timeout_s=None,
                record=None):

    if ctrl is None:
        controller = pid_next_waypoint_car_controller()
        logger.info('autodrive contoller was None, so was set to default {}'.format(ctrl.__class__))
    else:
        controller=ctrl # construct instance of the controller. The controllers car() is set later, once the server gives us the state

    if gui:
        launch_gui()
        args = get_args()
        game = client(track_name=args.track_name,
                      controller=ctrl,
                      spectate=args.spectate,
                      car_name=args.car_name,
                      server_host=args.host,
                      server_port=args.port,
                      joystick_number=args.joystick,
                      fps=args.fps,
                      timeout_s=args.timeout_s,
                      record=args.record)
    else:

        IGNORE_COMMAND = '--ignore-gooey'
        if IGNORE_COMMAND in sys.argv:
            sys.argv.remove(IGNORE_COMMAND)

        args = get_args()

        if args.record:
            infofile = write_args_info(args, 'data')

            fh = logging.FileHandler(infofile)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)


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

        game = client(track_name=track_name,
                      spectate=spectate,
                      car_name=car_name,
                      controller=ctrl,
                      server_host=server_host,
                      server_port=server_port,
                      joystick_number=joystick_number,
                      fps=fps,
                      timeout_s=timeout_s,
                      record=record)

    return game
