"""
Client l2race agent

"""

import sys
sys.path.insert(0, "src/")
sys.path.insert(1, "commonroad-vehicle-models/PYTHON/")

import importlib
import os
from typing import Tuple, List, Optional, Dict
import argparse
import argcomplete as argcomplete
import select
from easygui import easygui, fileopenbox, indexbox
from pandas import DataFrame
from pygame.math import Vector2
import pygame.freetype  # Import the freetype module.
import pickle
import socket
import time
import pygame
import sys
import atexit
import pandas as pd
import re
import timeit
# https://www.programiz.com/python-programming/shallow-deep-copy#:~:text=help%20of%20examples.-,Copy%20an%20Object%20in%20Python,reference%20of%20the%20original%20object.
import  copy
# import kivy
import time


from l2race_settings import *
from my_args import client_args, write_args_info
from car_command import car_command
from car_state import car_state
from data_recorder import data_recorder
from l2race_utils import find_unbound_port_in_range, open_ports, loop_timer
from track import track, list_tracks
from car import car, loadAndScaleCarImage
from l2race_utils import my_logger
from controllers import car_controller
from controllers.pid_next_waypoint_car_controller import pid_next_waypoint_car_controller
from models.models import linear_extrapolation_model, client_car_model
from keyboard_and_joystick_input import keyboard_and_joystick_input
from user_input import user_input

logger = my_logger(__name__)


# logger.setLevel(logging.DEBUG) # uncomment to debug

def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = client_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # check if show dialog to choose track
    if args.track_name is None or args.track_name=='' or str.lower(args.track_name)=='none' or str.lower(args.track_name)=='choose' or str.lower(args.track_name)=='dialog':
        args.track_name=select_track_dialog()
        if args.track_name is None:
            logger.info(f'user did not select any track')
            quit(0)
    return args


def select_track_dialog()->Optional[str]:
    """ Show dialog for track

    :returns: track name or None if cancelled
    """
    tracks = list_tracks()
    track_num = indexbox('What track do you want?', 'Tracks', tracks)
    if track_num is None:
        return None
    track_name = tracks[track_num]
    logger.info(f'User chose {track_name} by dialog')
    return track_name


class client:

    def __init__(self,
                 track_name: str = 'track',
                 spectate: bool = False,
                 car_name: str = CAR_NAME,
                 controller: car_controller = None,
                 client_car_model : client_car_model= None,
                 server_host: str = SERVER_HOST,
                 server_port: int = SERVER_PORT,
                 joystick_number: int = JOYSTICK_NUMBER,  # TODO not used below
                 fps: float = FPS,
                 widthPixels: int = SCREEN_WIDTH_PIXELS,
                 heightPixels: int = SCREEN_HEIGHT_PIXELS,
                 timeout_s: float = SERVER_TIMEOUT_SEC,
                 record: Optional[str] = None,
                 replay_file_list: Optional[List[str]] = None,
                 lidar=None  # TODO add type hint
                 ):
        """
        Makes a new instance of client that users use to run a car on a track.

        :param track_name: string name of track, without .png suffix
        :param spectate: set True to just spectate
        :param car_name: Your name for your car
        :param controller: Optional autodrive controller that implements the read method to return (car_command, user_input)
        :param client_car_model: Optional client_car_model that updates a car_state of our own
        :param server_host: hostname of server, e.g. 'localhost' or 'telluridevm.iniforum.ch'
        :param server_port: port on server to initiate communication
        :param joystick_number: joystick number if more than one
        :param fps: desired frames per second of game loop
        :param widthPixels:  screen width in pixels (must match server settings)
        :param heightPixels: height, must match server settings
        :param timeout_s: socket read timeout for blocking reads (main loop uses nonblocking reads)
        :param record: set it None to not record. Set it to a string to add note for this recording to file name to record data for all cars to CSV files
        :param replay_file_list: None for normal live mode, or List[str] of filenames to play back a set of car recordings together
        :param lidar: Optional lidar to display TODO what type is this argument?
        """

        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.spectate = spectate
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        pygame.freetype.init()
        pygame.game_font: pygame.freetype.SysFont =None
        try:
            self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        except:
            logger.warning('cannot get specified globals.py font {}, using pygame default font'.format(GAME_FONT_NAME))
            self.game_font = pygame.freetype.SysFont(pygame.font.get_default_font(), GAME_FONT_SIZE)
        self.ghost_car_running_font: pygame.font.Font=None
        self.clock = pygame.time.Clock()
        self.exit = False
        self.input = None
        self.fps = fps
        self.sock: Optional[socket] = None  # our socket used for communicating with server
        self.server_host: str = server_host
        self.server_port: int = server_port
        self.serverStartAddr: Tuple[str, int] = (
            self.server_host, self.server_port)  # manager address, different port on server used during game
        self.gameSockAddr: Optional[Tuple[str, int]] = None  # address used during game
        self.server_timeout_s: float = timeout_s
        self.gotServer: bool = False
        self.recording_enabled: bool = not record is None
        self.record_note: Optional[str] = record if not record is None else None
        self.data_recorders: Optional[List[data_recorder]] = None
        self.replay_file_list: Optional[List[str]] = replay_file_list
        self.track_name: str = track_name
        self.car_name: str = car_name
        self.car: Optional[car] = None  # will make it later after we get info from server about car
        self.input: keyboard_and_joystick_input = keyboard_and_joystick_input()
        self.car_command: car_command = car_command() # current car_command
        self.user_input: user_input = user_input() # current user_input
        self.autodrive_controller = controller  # automatic self driving controller specified in constructor
        self.client_car_model = client_car_model  # our model of car
        self.ghost_car: Optional[car] = None  # this is car that we show when running self.client_car_model

        self.lidar = lidar  # variable controlling if to show lidar mini and with what precission
        self.t_max = 0.0

        self.server_message = None  # holds messsages sent from server to be displayed
        self.last_server_message_time = time.time()

        # if controller is None:
        #     self.controller = pid_next_waypoint_car_controller()
        # else:
        #     self.controller = controller

        # spectator data structures
        self.track_instance: track = track(track_name=self.track_name)
        self.spectate_cars: Dict[
            str, car] = dict()  # dict of other cars (NOT including ourselves) on the track, by name of the car. Each entry is a car() that we make here. For spectators, the list contains all cars. The cars contain the car_state. The complete list of all cars is this dict plus self.car

    def cleanup(self):
        """
        Cleans up client before shutdown.
        :return: None
        """
        if self.gotServer:
            if self.spectate:
                self.send_to_server(self.gameSockAddr, 'remove_spectator', None)
            else:
                self.send_to_server(self.gameSockAddr, 'remove_car', self.car_name)
        if self.sock:
            self.sock.close()
            self.sock = None

    def render_multi_line(self, text, x, y, color=None):  # todo clean up
        """
        Renders a multiline string to the screen.
        :param text: some string with embedded \n
        :param x: x starting from left
        :param y: y starting from top
        :param color: Tuple[r,g,b] 0-255
        :return: None
        """
        if color is None:
            color = (200, 200, 200)
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.game_font.render_to(self.screen, (x, y + GAME_FONT_SIZE * i), l, color),
            pass

    def ping_server(self):
        logger.info('pinging server at {}'.format(self.serverStartAddr))
        self.drain_udp_messages()
        t = time.time()
        self.send_to_server(self.serverStartAddr, 'ping', None)
        try:
            (msg, payload) = self.receive_from_server(blocking=True)
        except socket.timeout:
            logger.warning('ping timeout')
            return False
        except ConnectionResetError:
            logger.warning('ping connection reset error')
            return False
        except Exception as e:
            logger.warning('ping other exception {}'.format(e))
            return False
        if msg != 'pong':
            logger.warning('wrong response {} received for ping'.format(msg))
            return False
        else:
            dt = time.time() - t
            logger.info('pong received with latency {:.1f}ms'.format(dt * 1000))
            return True

    def connect_to_server(self):
        """
        Connects to server  to start car on the track

        :return: None
        """
        if self.gotServer:
            return
        logger.info('connecting to l2race model server at ' + str(self.serverStartAddr) + ' to add car or spectate')
        ntries = 0
        looper = loop_timer(rate_hz=1. / SERVER_PING_INTERVAL_S)
        err_str = ''
        while not self.gotServer:
            looper.sleep_leftover_time()
            self.screen.fill([0, 0, 0])
            self.render_multi_line(err_str, 10, 10)
            pygame.display.flip()
            ntries += 1
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True  # TODO clean up, seems redundant. what sets pygame.QUIT?
                    break
            self.input.read(self.car_command, self.user_input )
            if self.user_input.quit:
                logger.info('startup aborted before connecting to server')
                pygame.quit()
            if not self.ping_server():
                err_str = 'No response to ping server at {}, will try again in {:.1f}s [{}]'. \
                    format(self.serverStartAddr, 1. / looper.rate_hz, looper.loop_counter)
                continue
            if not self.spectate:
                cmd = 'add_car'
                payload = (self.track_name, self.car_name)
            else:
                cmd = 'add_spectator'
                payload = self.track_name
            err_str = 'sending cmd={} with payload {} to server initial address {}, ' \
                      'waiting for server...[{}]' \
                .format(cmd, payload, self.serverStartAddr, ntries)
            logger.info(err_str)
            self.screen.fill([0, 0, 0])
            self.render_multi_line(err_str, 10, 10)
            pygame.display.flip()
            self.send_to_server(self.serverStartAddr, cmd, payload)

            try:
                # now get the game port as a response
                logger.info('pausing for server track process to start (if not running already)')
                time.sleep(
                    2)  # it takes significant time to start the track process. To avoid timeout, wait a bit here before checking
                logger.info('receiving game_port message from server')
                msg, payload = self.receive_from_server()
                if msg != 'game_port':
                    logger.warning(
                        "got response (msg,command)=({},{}) but expected ('game_port',port_number); will try again in {}s".format(
                            msg, payload, SERVER_PING_INTERVAL_S))
                    continue
            except OSError as err:
                err_str += '\n{}:\n error for response from {}; ' \
                           'will try again in {}s ...[{}]'.format(err, self.serverStartAddr, SERVER_PING_INTERVAL_S,
                                                                  ntries)
                logger.warning(err_str)
                continue
            port = int(payload)
            self.gotServer = True
            self.gameSockAddr: Tuple[str, int] = (self.server_host, port)
            logger.info('got game_port message from server telling us to use address {} to talk with server'.format(
                self.gameSockAddr))
            if not self.spectate:
                self.car = car(name=self.car_name, our_track=self.track_instance, screen=self.screen,
                               client_ip=self.gameSockAddr)

                if self.autodrive_controller:
                    self.autodrive_controller.car = self.car # note, shallow copy

                logger.info('initial car state is {}'.format(self.car.car_state))


    def run(self) -> None:
        """
        Either runs the game live or replays recording(s), depending on self.replay_file_list

        """
        if self.replay_file_list is not None:
            if self.replay():
                logger.info('Done replaying')
            else:
                logger.error('Could not replay file')
        else:
            self.run_new_game()

    def run_new_game(self) -> None:
        """
        Runs the game in live mode.

        """
        if self.server_host == 'localhost':
            logger.info('skipping opening ports for local server')
        else:
            try:
                self.render_multi_line(
                    'opening necessary UDP ports in CLIENT_PORT_RANGE {}...'.format(CLIENT_PORT_RANGE), 10, 10,
                    [200, 200, 200])
                pygame.display.flip()
                # open_ports()
            except Exception as ex:
                logger.warning("Caught exception '{}' when trying to open l2race client ports".format(ex))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.sock.settimeout(self.server_timeout_s)
        find_unbound_port_in_range(CLIENT_PORT_RANGE)

        atexit.register(self.cleanup)

        logger.info('starting main loop')
        looper = loop_timer(self.fps)
        while not self.exit:
            try:
                looper.sleep_leftover_time()
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt, stopping client')
                self.exit = True

            # Drawing
            self.draw()
            pygame.display.flip()

            self.connect_to_server()

            # User input TODO move to method to get user or autodrive input
            self.process_user_or_autodrive_input()

            # TODO move to method that uses select to check for response if any
            # expect to get new car state
            try:
                cmd, payload = self.receive_from_server(blocking=False)
                if cmd is None:
                    continue

                self.handle_message(cmd, payload)

                # toggle recording of ourselves
                if self.recording_enabled:
                    self.start_recording()
                    self.maybe_record_data()
                else:
                    self.stop_recording()

            except socket.timeout:
                logger.warning('Timeout on socket receive from server, using previous car state. '
                               'Check server to make sure it is still running')
            except pickle.UnpicklingError as err:
                logger.warning('{}: could not unpickle the response from server'.format(err))
            except TypeError as te:
                logger.warning(str(te) + ": ignoring and waiting for next state")
            except ConnectionResetError:
                logger.warning('Connection to {} was reset, will look for server again'.format(self.gameSockAddr))
                self.gotServer = False

            self.update_ghost_car()

        logger.info('ending main loop')
        self.cleanup()
        logger.info('quitting pygame')
        pygame.quit()
        quit()

    def stop_recording(self):
        if self.data_recorders:
            for r in self.data_recorders:
                r.close_recording()
                r = None
            self.data_recorders = None

    def maybe_record_data(self):
        if self.data_recorders:
            for r in self.data_recorders:
                r.write_sample()

    def process_user_or_autodrive_input(self):
        """
        Gets user or agent input and sends to model server, then does a non-blocking read
        to see if there is any message from server.

        :return: (cmd,payload) available from server, or None,None if nothing is available.
        """
        self.input.read(self.car_command,self.user_input)
        if self.car_command.autodrive_enabled:
            if self.autodrive_controller is None:
                raise RuntimeError(
                    'Tried to use autodrive control but there is no controller defined. See AUTODRIVE_CLASS in src/globals.py or on command line with --autodrive.')
            self.autodrive_controller.read(self.car_command)

        if self.user_input.quit:
            logger.info('quit recieved from keyboard or joystick, ending main loop')
            self.exit = True
            return

        if self.user_input.toggle_recording:
            self.recording_enabled = not self.recording_enabled
            logger.info(f'toggled recording_enabled={self.recording_enabled}')

        if self.user_input.open_playback_recording:
            file=fileopenbox(msg='Select .csv recording file',
                                         title='l2race',
                                         filetypes=[['*.csv','Comma Separated Values file']],
                                         multiple=False,
                                         default='*.csv')
            logger.info(f'selected {file} with file dialog')
            self.replay(file)

        if self.user_input.choose_new_track:
            name=select_track_dialog()
            if name is not None:
                self.track_name=name
                self.__init__(car_name=self.car_name,
                              track_name=self.track_name,
                              controller=self.autodrive_controller,
                              client_car_model=self.client_car_model,
                              server_host=self.server_host,
                              server_port=self.server_port)
                self.run()
            pass

        if not self.spectate:
            if self.user_input.restart_car:
                self.restart_car('user asked to restart car')

            if self.user_input.restart_client:
                logger.info('restarting client')
                self.gotServer = False
                if self.data_recorders:
                    for r in self.data_recorders:
                        r.close_recording()
                    self.data_recorders = None
            if self.user_input.choose_new_track:
                pass
            # send control to server
            self.send_to_server(self.gameSockAddr, 'command', self.car_command)
        else:
            self.send_to_server(self.gameSockAddr, 'send_states', None)

    def receive_from_server(self, blocking=False) -> Tuple[Optional[str], Optional[object]]:
        ''' attempt to receive msg from server
        :param blocking - set true for blocking receive. If false, returns None,None if there is nothing for us
        :returns (cmd,payload), or None,None if nonblocking and nothing is ready
        '''
        if not blocking:
            inputready, o, e = select.select([self.sock], [], [], 0.0)
            if len(inputready) == 0: return None, None  # nothing for us now
        data, server_addr = self.sock.recvfrom(
            8192)  # todo check if large enough for payload inclding all other car state
        (cmd, payload) = pickle.loads(data)
        logger.debug('got message {} with payload {} from server {}'.format(cmd, payload, server_addr))
        return cmd, payload

    def send_to_server(self, addr: Tuple[str, int], msg: str, payload: object):
        """
        Send msg,payload to server at specified (ip,port)

        :param addr: (ip,port) of track port
        :param msg: the string message type
        :param payload: the payload object

        :return: None

        """
        if self.sock is None:
            logger.warning('no socket to send message {} with payload {}'.format(msg, payload))
            return
        logger.debug('sending msg {} with payload {} to {}'.format(msg, payload, addr))
        p = pickle.dumps((msg, payload))
        self.sock.sendto(p, addr)

    def process_top_ten_list(self, payload):
        pass

    def ask_for_top_ten_list(self):
        pass

    def drain_udp_messages(self):
        """remove the data present on the socket. From https://stackoverflow.com/questions/1097974/how-to-empty-a-socket-in-python """
        logger.debug('draining existing received UDP messages')
        sock_input = [self.sock]
        try:
            while True:
                inputready, o, e = select.select(sock_input, [], [], 0.0)
                if len(inputready) == 0: break
                for s in inputready: s.recv(8192)
        except Exception as e:
            logger.warning('caught {} when draining received UDP port messaages'.format(e))

    def draw(self):
        """
        Top level drawing command. Call pygame.display.flip() after other drawing commands after draw().
        :return: None
        """
        lidar = False
        self.track_instance.draw(self.screen)
        self.draw_other_cars()
        self.draw_server_message()
        self.draw_own_car()
        self.draw_ghost_car()

        self.draw_lidar()

    def draw_own_car(self):
        if self.car:
            self.car.draw(self.screen)
            self.render_multi_line(str(self.car.car_state), 10, 10)

    def draw_ghost_car(self):
        """
        Draws the ghost car view of our car model
        """
        if self.ghost_car and self.user_input.run_client_model:
            self.ghost_car.draw(self.screen)
        if self.user_input.run_client_model:
            # show it prominently in display
            size=24
            if self.ghost_car_running_font is None:
                try:
                    self.ghost_car_running_font = pygame.freetype.SysFont(GAME_FONT_NAME, size=size)
                except:
                    logger.warning('cannot get specified globals.py font {}, using pygame default font'.format(GAME_FONT_NAME))
                    self.ghost_car_running_font = pygame.freetype.SysFont(pygame.font.get_default_font(), size=size)

            self.ghost_car_running_font.render_to(surf=self.screen, dest=(10, self.screen.get_height()-10-self.ghost_car_running_font.get_sized_height()), text='Ghost car model is running', fgcolor=(255,0,0))


    def draw_other_cars(self):
        """ Draws all the others"""
        for c in self.spectate_cars.values():
            c.draw(self.screen)

    def draw_server_message(self):
        """ Draws message from server, if any """
        if self.server_message is None:
            return
        if time.time() - self.last_server_message_time > 10:
            self.server_message = None
            return
        if self.server_message.startswith('ERROR'):
            color = (255, 10, 10)
        else:
            color = None
        self.render_multi_line(str(self.server_message), 10, SCREEN_HEIGHT_PIXELS - 50, color=color)

    def draw_lidar(self) -> None:
        if self.lidar and self.car is not None:
            # t0 = timeit.default_timer()
            x_track = self.car.car_state.position_m.x
            y_track = self.car.car_state.position_m.y
            x_map = self.track_instance.meters2pixels(x_track)
            y_map = self.track_instance.meters2pixels(y_track)
            hit_pos = self.track_instance.find_hit_position(angle=self.car.car_state.body_angle_deg,
                                                            pos=(x_map, y_map),
                                                            dl=self.lidar)
            if hit_pos is not None:
                pygame.draw.line(self.screen, (0, 0, 255), (x_map, y_map), hit_pos)
                pygame.draw.circle(self.screen, (0, 255, 0), hit_pos, 3)
            # t1 = timeit.default_timer()
            # t = t1-t0
            # if t>self.t_max:
            #     self.t_max=t
            #     print(t)

    def update_state(self, all_states: List[car_state]) -> None:
        """
        Updates list of internal state.

        :param all_states: the list of states of all cars from model server
        :return: None
        """
        # # make a list of all the cars in our list of spectate_cars
        # plus our own car that are not in the state list we just got
        current_state_car_names = []
        for s in all_states:
            current_state_car_names.append(s.static_info.name)
        dict_car_names = []
        for s in self.spectate_cars.keys():
            dict_car_names.append(s)
        if self.car:
            dict_car_names.append(self.car.car_state.static_info.name)
        to_remove: List[str] = [x for x in dict_car_names if x not in current_state_car_names]
        dr_to_remove = []
        for r in to_remove:
            del self.spectate_cars[r]
            for dr in self.data_recorders:
                if dr.car.name() == r:
                    logger.debug('closing data recorder for lost car {}'.format(r))
                    dr.close_recording()
                    dr_to_remove.append(dr)
        for d in dr_to_remove:
            del self.data_recorders[d]
        for s in all_states:
            name = s.static_info.name  # get the car name from the remote state
            if name == self.car_name:
                self.car.car_state = s  # update our own state
                if self.car.car_state.check_if_went_haywire() is not None:
                    logger.error(f'car state went untable, restarting car')
                    self.restart_car()
                    return
                continue  # don't add ourselves to list of other (spectator) cars
            # update other cars on the track
            c = self.spectate_cars.get(name)  # get the car
            if c is None:  # if it doesn't exist, construct it
                self.spectate_cars[name] = car(name=name,
                                               image_name='car_other.png',
                                               our_track=self.track_instance,
                                               client_ip=s.static_info.client_ip,
                                               screen=self.screen)
            self.spectate_cars[name].car_state = s  # set its state

        # manage recordings
        recording_car_list = []
        if self.data_recorders:
            for d in self.data_recorders:  # make list of car names we are recording
                recording_car_list.append(d.car.name())
        current_other_car_list = []
        for c in self.spectate_cars.keys():  # make all current cars
            current_other_car_list.append(c)
        if self.recording_enabled:
            # find current cars that we are not yet recording
            not_yet_recording = [x for x in current_other_car_list if x not in recording_car_list]
            for c in not_yet_recording:
                sc = self.spectate_cars[c]
                if sc.car_state.hostname() == self.car.car_state.hostname():
                    continue  # don't record other cars running also from us
                dr = data_recorder(car=sc)
                self.data_recorders.append(dr)
                try:
                    dr.open_new_recording()
                except RuntimeError as e:
                    logger.warning('Could not open data recorder for car {}: caught exception {}'.format(sc, e))
        # logger.debug('After update, have own car {} and other cars {}'.format(self.car.car_state.static_info.name if self.car else 'None', self.spectate_cars.keys()))

    def handle_message(self, msg: str, payload: object) -> None:
        """
        Handle message from model server.

        :param msg: the msg str
        :param payload: the payload object
        :return: None
        """
        if msg == 'state':
            self.update_state(payload)  # assumes that payload is List[car_state]
        elif msg == 'game_port':
            self.gameSockAddr = (self.server_host, payload)
        elif msg == 'track_shutdown':
            logger.warning('{}, will try to look for it again'.format(payload))
            self.gotServer = False
        elif msg == 'string_message':
            logger.info('recieved message "{}"'.format(payload))
            self.last_server_message_time = time.time()
            self.server_message = payload
        else:
            logger.warning(
                'unexpected msg {} with payload {} received from server (should have gotten "car_state" message)'.format(
                    msg, payload))

    def replay(self, filename=None) -> bool:
        """
        Replays the self.replay_file_list recordings. It will immediately return False if it cannot find the file to play. Otherwise
        it will start a loop that runs over the entire file to play it, and finally return True.

        :param file: CSV file to play back

        :returns: False if it cannot find the file to play, True at the end of playing the entire recording.
        """
        # Load data
        # Find the right file
        if filename is not None:
            file_path=filename
        elif (self.replay_file_list is not None) and (self.replay_file_list != 'last'):

            if isinstance(self.replay_file_list, List) and len(self.replay_file_list) > 1:
                filename = self.replay_file_list[0]
                raise NotImplemented('Replaying more than one recording is not yet implemented')

            if isinstance(self.replay_file_list, str):
                filename = self.replay_file_list

            if not filename.endswith('.csv'):
                filename = filename + '.csv'

            # if filename is a path, then use the path, otherwise, look for file at starting folder and in DATA_FOLDER_NAME folder
            if os.sep in filename:
                # treat as local or DATA_FOLDER filename
                file_path = filename
                if not os.path.isfile(filename):
                    logger.error('Cannot replay: There is no race recording file with name {}'.format(filename))
                    return False
            else:
                # check if file found in DATA_FOLDER_NAME or at local starting point
                if not os.path.isfile(filename):
                    file_path = os.path.join(DATA_FOLDER_NAME, filename)
                    if not os.path.isfile(file_path):
                        logger.error(
                            'Cannot replay: There is no race recording file with name {} at local folder or in {}'.format(
                                file_path, DATA_FOLDER_NAME))
                        return False
                else:
                    file_path = filename

        elif self.replay_file_list == 'last':
            try:
                import glob
                list_of_files = glob.glob(DATA_FOLDER_NAME + '/*.csv')
                file_path = max(list_of_files, key=os.path.getctime)
            except FileNotFoundError:
                logger.error('Cannot replay: No race recording found in data folder ' + DATA_FOLDER_NAME)
                return False
        else:
            logger.error('Cannot replay: filename is None')
            return False

        # Get race recording
        logger.info(f'Replaying file {file_path}')
        try:
            data: DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
        except Exception as e:
            logger.error('Cannot replay: Caught {} trying to read CSV file {}'.format(e, file_path))
            return False

        def get_header_params(filename):
            para_dic = {}
            with  open(filename,'r') as cmt_file:    # open file
                for line in cmt_file:    # read each line
                    if line[0] == '#':    # check the first character
                        line = line[1:]    # remove first '#'
                        para = line.split('=')     # seperate string by '='
                        if len(para) == 2:
                            para_dic[para[0].strip()] = para[1].strip().strip("\"") # remove " from around param
                    else:
                        break # done with header
            return para_dic

        header_params=get_header_params(file_path)
        logger.info('header field parameters from {} are \n{}'.format(file_path,header_params))

        # Get used car name
        track_name = header_params['track_name']
        logger.info('CSV file track is named '+track_name)
        if track_name!=self.track_name:
            logger.warning('replay track name "{}" differs from command line defined track name "{}", making new track instance for replay'.format(track_name,self.track_name))
            self.track_name=track_name
            self.track_instance: track = track(track_name=self.track_name)


        # Get used track
        self.car_name = header_params['car_name']
        logger.debug('CSV file car is named' + self.car_name)

        # Define car and track
        self.track_instance = track(self.track_name)
        self.car = car(name=self.car_name, our_track=self.track_instance, screen=self.screen)

        # decimate data to make it play faster
        # data = data.iloc[::4, :]

        # Run a loop to print data
        n_rows = data.shape[0]
        r = 0
        step = 1
        scale = 10
        looper = loop_timer(rate_hz=self.fps)
        while not self.exit:
            try:
                looper.sleep_leftover_time()
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt, stopping client')
                self.exit = True

            self.input.read(self.car_command, self.user_input)  # gets input from keyboard or joystick
            if self.user_input.quit:
                self.exit = True
                continue
            playback_speed = self.car_command.steering

            row:dict = data.iloc[r]  # position based indexing of DataFrame https://pythonhow.com/accessing-dataframe-columns-rows-and-cells/
            self.car.car_state.parse_csv_row(row)

            self.update_ghost_car()

            # Drawing
            self.draw()
            frac = float(r) / n_rows
            w = frac * (self.screen.get_width() - 10)
            pygame.draw.rect(self.screen, [200, 200, 200], [10, self.screen.get_height() - 20, w, 10], True)

            pygame.display.flip()

            if self.user_input.quit:
                self.cleanup()
                break

            if self.user_input.close_playback_recording:
                self.restart_car()
                self.run_new_game()

            if self.user_input.restart_car:
                r = 0
            if playback_speed >= -0.05:  # offset from zero to handle joysticks that have negative offset
                r = r + step if r < n_rows - step else n_rows
                if r > n_rows - 1: r = n_rows - 1
            else:
                r = r - step if r > 0 else 0
            step = int(abs(playback_speed) * scale)
            step = 1 if step < 1 else step
            # speedup is factor times normal speed, limited by rendering rate
            looper.rate_hz = self.fps * (1 + abs(scale * playback_speed))
        return True

    def restart_car(self, message: str = None) -> None:
        """
        Request server to restart the car.
        :param message: message that server will log
        :returns: None
        """
        # car restart handled on server side
        self.stop_recording()
        logger.info('sending message to restart car to server')
        self.send_to_server(self.gameSockAddr, 'restart_car', message)
        self.car_command=car_command() # override in case server does not reset the autodrive_enabled flag TODO not clear why needed
        if self.recording_enabled:
            self.start_recording()


    def update_ghost_car(self) -> None:
        """
        updates the 'ghost' client car model if we there is one and the mode is enabled
        """

        if self.client_car_model is None:
            logger.warning('tried to run client_car_model as ghost car but client_car_model is None')
            return

        if self.ghost_car is None :
            if self.car is None or self.car.car_state is None:
                return  # we might be spectating, so don't have a car to ghost
            logger.info('making ghost car copy of car for showing client_car_model')
            self.ghost_car = copy.copy(self.car)  # just copy fields
            self.ghost_car.car_state = copy.copy(self.car.car_state)  # make sure we get our own car_state
            self.ghost_car.car_state.command=copy.copy(self.car.car_state.command) # and our own command
            self.ghost_car.image_name = 'ghost_car'
            self.ghost_car.image=loadAndScaleCarImage(self.ghost_car.image_name, self.ghost_car.car_state.static_info.length_m, self.screen)

        self.client_car_model.update_state(self.user_input.run_client_model, self.car.car_state.time, self.car_command, self.car, self.ghost_car)

    def start_recording(self):
        if self.data_recorders is None:  # todo add other cars to data_recorders as we get them from server
            self.data_recorders = [data_recorder(car=self.car)]
            try:
                self.data_recorders[0].open_new_recording()
            except RuntimeError as e:
                logger.warning('Could not open data recording; caught {}'.format(e))

def main():
    args = get_args()
    try:
        mod = importlib.import_module(args.autodrive[0])
        cl=getattr(mod, args.autodrive[1])
        controller = cl() # set it to a class in globals.py
        logger.info('using autodrive controller {}'.format(controller))
    except Exception as e:
        logger.error('cannot import AUTODRIVE_CLASS named "{}" from module AUTODRIVE_MODULE named "{}", got exception {}'.format(args.autodrive[1], args.autodrive[0],e))
        controller=None

    try:
        mod = importlib.import_module(args.carmodel[0])
        cl=getattr(mod, args.carmodel[1])
        car_model = cl() # set it to a class in globals.py
        logger.info('using client car_model {}'.format(car_model))
    except Exception as e:
        logger.error('cannot import CAR_MODEL_CLASS named "{}" from module CAR_MODEL_MODULE named "{}", got exception {}'.format(args.carmodel[1], args.carmodel[0],e))
        car_model=None

    if not args.record is None:  # if recording data, also record command line arguments and log output to a text file
        timestr = time.strftime("%Y%m%d-%H%M")
        filepath = os.path.join(DATA_FOLDER_NAME, 'l2race-log-' + str(timestr) + '.txt')
        logger.info('Since recording is enabled, writing arguments and logger output to {}'.format(filepath))

        infofile = write_args_info(args, filepath)

        fh = logging.FileHandler(infofile)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if args.car_name is None:  # If car_name not provided as a terminal-line argument...
        try:  # try to create a car_name based on the host name
            # make hopefully unique car name
            import socket, getpass, random, string
            hostname = socket.gethostname()
            username = getpass.getuser()
            car_name = str(hostname) + ':' + str(username) + '-'
            car_name += ''.join(random.choices(string.ascii_uppercase,
                                               k=2))  # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
        except:  # If nothing else works give it a default name
            car_name = CAR_NAME
    else:  # If car_name provided as terminal-line argument give it precedence
        car_name = args.car_name


    game = client(track_name=args.track_name,
                  controller=controller,
                  client_car_model=car_model,
                  spectate=args.spectate,
                  car_name=car_name,
                  server_host=args.host,
                  server_port=args.port,
                  joystick_number=args.joystick,
                  fps=args.fps,
                  timeout_s=args.timeout_s,
                  record=args.record,
                  replay_file_list=args.replay,
                  lidar=args.lidar)
    game.run()

if __name__ == '__main__':
    main()


