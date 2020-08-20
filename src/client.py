"""
Client l2race agent

"""
import os
from typing import Tuple, List, Optional, Dict
import argparse
import argcomplete as argcomplete
import select
from pygame.math import Vector2
import pygame.freetype  # Import the freetype module.
import pickle
import socket
import time
from time import sleep
import pygame
import sys
import atexit
import pandas as pd
import re

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

logger = my_logger(__name__)


# logger.setLevel(logging.DEBUG) # uncomment to debug

def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        epilog='Run with no arguments to open dialog for arguments, if Gooey is installed', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = client_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def launch_gui():
    # may only apply to windows
    try:
        from gooey import Gooey  # pip install Gooey
    except Exception:
        logger.info('Gooey GUI builder not available, will use command line arguments.\n'
                    'Install with "pip install Gooey". See README')

    try:
        ga = Gooey(get_args, program_name="l2race client", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.info('Gooey GUI not available, using command line arguments. \n'
                    'You can try to install with "pip install Gooey". '
                    'Ignore this warning if you do not want a GUI launcher.')


class client:

    def __init__(self,
                 track_name:str='track',
                 spectate:bool=False,
                 car_name:str=CAR_NAME,
                 controller:Optional[object]=None,
                 server_host:str=SERVER_HOST,
                 server_port:int=SERVER_PORT,
                 joystick_number:int=JOYSTICK_NUMBER,
                 fps:float=FPS,
                 widthPixels:int=SCREEN_WIDTH_PIXELS,
                 heightPixels:int=SCREEN_HEIGHT_PIXELS,
                 timeout_s:float=SERVER_TIMEOUT_SEC,
                 record:Optional[str]=None,
                 replay_file_list:Optional[List[str]]=None
                 ):
        """
        Makes a new instance of client that users use to run a car on a track.

        :param track_name: string name of track, without .png suffix
        :param spectate: set True to just spectate
        :param car_name: Your name for your car
        :param controller: Optional autodrive controller that implements the read method to return (car_command, user_input)
        :param server_host: hostname of server, e.g. 'localhost' or 'telluridevm.iniforum.ch'
        :param server_port: port on server to initiate communication
        :param joystick_number: joystick number if more than one
        :param fps: desired frames per second of game loop
        :param widthPixels:  screen width in pixels (must match server settings)
        :param heightPixels: height, must match server settings
        :param timeout_s: socket read timeout for blocking reads (main loop uses nonblocking reads)
        :param record: set it None to not record. Set it to a string to add note for this recording to file name to record data for all cars to CSV files
        :param replay_file_list: None for normal live mode, or List[str] of filenames to play back a set of car recordings together
        """

        pygame.init()
        logger.info('using pygame version {}'.format(pygame.version.ver))
        pygame.display.set_caption("l2race")
        self.spectate = spectate
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.screen = pygame.display.set_mode(size=(self.widthPixels, self.heightPixels), flags=0)
        pygame.freetype.init()
        try:
            self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
        except:
            logger.warning('cannot get specified globals.py font {}, using pygame default font'.format(GAME_FONT_NAME))
            self.game_font = pygame.font.Font(pygame.font.get_default_font(), GAME_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.exit = False
        self.input = None
        self.fps = fps
        self.sock: Optional[socket] = None  # our socket used for communicating with server
        self.server_host: str = server_host
        self.server_port: int = server_port
        self.serverStartAddr: Tuple[str, int] = (self.server_host, self.server_port)  # manager address, different port on server used during game
        self.gameSockAddr: Optional[Tuple[str, int]] = None  # address used during game
        self.server_timeout_s:float = timeout_s
        self.gotServer:bool = False
        self.recording_enabled: bool = not record is None
        self.record_note:Optional[str]= record if not record is None else None
        self.data_recorders: Optional[List[data_recorder]] = None
        self.replay_file_list:Optional[List[str]]=replay_file_list
        self.track_name: str = track_name
        self.car_name: str = car_name
        self.car: Optional[car] = None  # will make it later after we get info from server about car
        try:
            self.input = my_joystick(joystick_number)
        except:
            self.input = my_keyboard()

        self.server_message = None  # holds messsages sent from server to be displayed
        self.last_server_message_time = time.time()

        # if controller is None:
        #     self.controller = pid_next_waypoint_car_controller()
        # else:
        #     self.controller = controller

        # spectator data structures
        self.track_instance: track = track(track_name=self.track_name)
        self.spectate_cars: Dict[str, car] = dict()  # dict of other cars (NOT including ourselves) on the track, by name of the car. Each entry is a car() that we make here. For spectators, the list contains all cars. The cars contain the car_state. The complete list of all cars is this dict plus self.car
        self.autodrive_controller = controller  # automatic self driving controller specified in constructor



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
            car_command, user_input = self.input.read()
            if user_input.quit:
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
                time.sleep(2)  # it takes significant time to start the track process. To avoid timeout, wait a bit here before checking
                logger.info('receiving game_port message from server')
                msg, payload = self.receive_from_server()
                if msg != 'game_port':
                    logger.warning("got response (msg,command)=({},{}) but expected ('game_port',port_number); will try again in {}s".format(msg, payload, SERVER_PING_INTERVAL_S))
                    continue
            except OSError as err:
                err_str += '\n{}:\n error for response from {}; ' \
                           'will try again in {}s ...[{}]'.format(err, self.serverStartAddr, SERVER_PING_INTERVAL_S, ntries)
                logger.warning(err_str)
                continue
            port = int(payload)
            self.gotServer = True
            self.gameSockAddr: Tuple[str, int] = (self.server_host, port)
            logger.info('got game_port message from server telling us to use address {} to talk with server'.format(self.gameSockAddr))
            if not self.spectate:
                self.car = car(name=self.car_name, track=self.track_instance, screen=self.screen, client_ip=self.gameSockAddr)
                if self.recording_enabled:
                    if self.data_recorders is None:  # todo add other cars to data_recorders as we get them from server
                        self.data_recorders = [data_recorder(car=self.car)]
                    self.data_recorders[0].open_new_recording()
                self.autodrive_controller.car = self.car
                logger.info('initial car state is {}'.format(self.car.car_state))

    def run(self)->None:
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

    def run_new_game(self)->None:
        """
        Runs the game in live mode.

        """
        if self.server_host == 'localhost':
            logger.info('skipping opening ports for local server')
        else:
            try:
                self.render_multi_line('opening necessary UDP ports in CLIENT_PORT_RANGE {}...'.format(CLIENT_PORT_RANGE), 10, 10, [200, 200, 200])
                pygame.display.flip()
                open_ports()
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
            if looper.loop_counter % CHECK_FOR_JOYSTICK_INTERVAL == 0 and not isinstance(self.input, my_joystick):
                try:
                    self.input = my_joystick()  # check for joystick that might get turned on during play
                except:
                    pass

            # Drawing
            self.draw()

            self.connect_to_server()
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input TODO move to method to get user or autodrive input
            self.process_user_or_autodrive_input()

            # TODO move to method that uses select to check for response if any
            # expect to get new car state
            try:
                cmd, payload = self.receive_from_server(blocking=False)
                if cmd is None:
                    continue
                self.handle_message(cmd, payload)
                if self.data_recorders:
                    for r in self.data_recorders:
                        r.write_sample()

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

        logger.info('ending main loop')
        self.cleanup()
        logger.info('quitting pygame')
        pygame.quit()
        quit()

    def process_user_or_autodrive_input(self):
        """
        Gets user or agent input and sends to model server, then does a non-blocking read
        to see if there is any message from server.

        :return: (cmd,payload) available from server, or None,None if nothing is available.
        """
        car_command, user_input = self.input.read()
        if car_command.autodrive_enabled:
            if self.autodrive_controller is None:
                raise RuntimeError('Tried to use autodrive control but there is no controller defined. See AUTODRIVE_CLASS in src/globals.py.')
            command = self.autodrive_controller.read()
        else:
            command = car_command

        if user_input.quit:
            logger.info('quit recieved, ending main loop')
            self.exit = True
            return

        if not self.spectate:
            if user_input.restart_car:
                self.restart_car('user asked to restart car')

            if user_input.restart_client:
                logger.info('restarting client')
                self.gotServer = False
                if self.data_recorders:
                    self.data_recorders.close_recording()

            # send control to server
            self.send_to_server(self.gameSockAddr, 'command', command)
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
        data, server_addr = self.sock.recvfrom(8192)  # todo check if large enough for payload inclding all other car state
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
        input = [self.sock]
        try:
            while True:
                inputready, o, e = select.select(input, [], [], 0.0)
                if len(inputready) == 0: break
                for s in inputready: s.recv(8192)
        except Exception as e:
            logger.warning('caught {} when draining received UDP port messaages'.format(e))

    def draw(self):
        """
        Top level drawing command.
        :return: None
        """
        self.track_instance.draw(self.screen)
        self.draw_other_cars()
        self.draw_server_message()
        self.draw_own_car()
        pygame.display.flip()

    def draw_own_car(self):
        if self.car:
            self.car.draw(self.screen)
            self.render_multi_line(str(self.car.car_state), 10, 10)

    def draw_other_cars(self):
        ''' Draws all the others'''
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

    def update_state(self, all_states: List[car_state]):
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
        to_remove = [x for x in dict_car_names if x not in current_state_car_names]
        for r in to_remove:
            del self.spectate_cars[r]
        for s in all_states:
            name = s.static_info.name  # get the car name from the remote state
            if name == self.car_name:
                self.car.car_state = s  # update our own state
                continue  # don't add ourselves to list of other (spectator) cars
            # update other cars on the track
            c = self.spectate_cars.get(name)  # get the car
            if c is None:  # if it doesn't exist, construct it
                self.spectate_cars[name] = car(name=name,
                                               image_name='car_other.png',
                                               track=self.track_instance,
                                               client_ip=s.static_info.client_ip,
                                               screen=self.screen)
            self.spectate_cars[name].car_state = s  # set its state
        # todo update list of data recorders if recording is enabled, close recordings for cars that left and remove them, add new cars to list of recorders

        # logger.debug('After update, have own car {} and other cars {}'.format(self.car.car_state.static_info.name if self.car else 'None', self.spectate_cars.keys()))

    def manage_data_recorders(self):
        """
        Maintains list of data_recorders to keep them in sync with state

        :return: None
        """
        pass

    def handle_message(self, msg: str, payload: object):
        """
        Handle message from model server.

        :param msg: the msg str
        :param payload: the payload object
        :return: None
        """
        if msg == 'state':
            self.update_state(payload)
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
            logger.warning('unexpected msg {} with payload {} received from server (should have gotten "car_state" message)'.format(msg, payload))

    def replay(self)->bool:
        """
        Replays the self.replay_file_list recordings. It will immediately return False if it cannot find the file to play. Otherwise
        it will start a loop that runs over the entire file to play it, and finally return True.

        :returns: False if it cannot find the file to play, True at the end of playing the entire recording.
        """
        # Load data

        # Find the right file
        if (self.replay_file_list is not None) and (self.replay_file_list is not 'last'):

            if isinstance(self.replay_file_list,List) and len(self.replay_file_list>1):
                raise NotImplemented('replaying more than one recording is not yet implemented')
            filename=self.replay_file_list[0]
            if not filename.endswith('.csv'):
                filename=filename+'.csv'
                file_path = os.path.join(DATA_FOLDER_NAME,filename)
                if not os.path.exists(file_path):
                    logger.warning('There is no race recording with name {}'.format(file_path))
                    return False

        elif self.replay_file_list=='last':
            try:
                import glob
                import os
                list_of_files = glob.glob('./data/*.csv')
                file_path = max(list_of_files, key=os.path.getctime)
            except FileNotFoundError:
                logger.warning('No race recording found in data folder')
                return False
        else:
            logger.warning('Program entered replay mode although race_name is None')
            return False

        # Get race recording
        logger.debug('replaying file {}'.format(file_path))
        data = pd.read_csv(file_path, comment='#') # skip comment lines starting with #

        # Get used car name
        s = str(pd.read_csv(file_path, skiprows=5, nrows=1))
        self.track_name = re.search('"(.*)"', s).group(1)
        logger.debug(self.track_name)

        # Get used track
        s = str(pd.read_csv(file_path, skiprows=6, nrows=1))
        self.car_name = re.search('"(.*)"', s).group(1)
        logger.debug(self.car_name)

        # print(file_path)
        # print(data.head())

        # Define car and track
        self.track_instance=track(self.track_name)
        self.car = car(name=self.car_name, track=self.track_instance, screen=self.screen)

        # decimate data to make it play faster
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
            self.car.car_state.velocity_m_per_sec = Vector2(row['vel.x'], row['vel.y'])
            self.car.car_state.speed_m_per_sec = row['speed']
            self.car.car_state.accel_m_per_sec_2 = Vector2(row['accel.x'], row['accel.y'])
            self.car.car_state.steering_angle_deg = row['steering_angle']
            self.car.car_state.body_angle_deg = row['body_angle']
            self.car.car_state.yaw_rate_deg_per_sec = row['yaw_rate']
            self.car.car_state.drift_angle_deg = row['drift_angle']

            # Drawing
            self.draw()
            sleep(1 / self.fps)
        return True

    def restart_car(self, message: str = None):
        """
        Request server to restart the car.
        :param message: message that server will log
        :returns: None
        """
        # car restart handled on server side
        logger.info('sending message to restart car to server')
        self.send_to_server(self.gameSockAddr, 'restart_car', message)


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
                record=False,
                record_note=None,
                replay=None) -> client:
    """
    Defines the client side of l2race game for user by handling the command line arguments if they are supplied.

        :param gui: Set true to use Gooey to put up dialog to launch client, if Gooey is installed.
        :param track_name: string name of track, without .png suffix
        :param spectate: set True to just spectate
        :param car_name: Your name for your car
        :param controller: Optional autodrive controller that implements the read method to return (car_command, user_input)
        :param server_host: hostname of server, e.g. 'localhost' or 'telluridevm.iniforum.ch'
        :param server_port: port on server to initiate communication
        :param joystick_number: joystick number if more than one
        :param fps: desired frames per second of game loop
        :param widthPixels:  screen width in pixels (must match server settings)
        :param heightPixels: height, must match server settings
        :param timeout_s: socket read timeout for blocking reads (main loop uses nonblocking reads)
        :param record: boolean, set it True to record data for all cars to CSV files
        :param record_note: note to add to recording CSV filename and header section
        :param replay: None for normal live mode, or List[str] of filenames to play back a set of car recordings together

    :return: the client(), which must them be started.
    """
    if ctrl is None:
        controller = pid_next_waypoint_car_controller()
        logger.info('autodrive contoller was None, so was set to default {}'.format(ctrl.__class__))
    else:
        controller = ctrl  # construct instance of the controller. The controllers car() is set later, once the server gives us the state

    args = get_args()
    if not args.record is None: # if recording data, also record command line arguments and log output to a text file
        import time
        timestr = time.strftime("%Y%m%d-%H%M")
        filepath = os.path.join(DATA_FOLDER_NAME, 'l2race-log-'+str(timestr)+'.txt')
        logger.info('Since recording is enabled, writing arguments and logger output to {}'.format(filepath))

        infofile = write_args_info(args, filepath)

        fh = logging.FileHandler(infofile)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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
                      record=args.record,
                      replay_file_list=args.replay)
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

        if replay is None:
            replay=args.replay

        game = client(track_name=track_name,
                      spectate=spectate,
                      car_name=car_name,
                      controller=ctrl,
                      server_host=server_host,
                      server_port=server_port,
                      joystick_number=joystick_number,
                      fps=fps,
                      timeout_s=timeout_s,
                      record=args.record,
                      replay_file_list=args.replay)

    return game
