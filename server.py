import argparse
import socket, pickle
import threading
import argcomplete
from timeit import default_timer as timer
from time import sleep
import multiprocessing as mp

from src.l2race_utils import bind_socket_to_range
from src.my_args import server_args
from src.car_model import CarModel
from src.car import car
from src.globals import *
from src.track import track, list_tracks
# may only apply to windows
from src.my_logger import my_logger
logger=my_logger(__name__)
try:
    from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception:
    logger.warning('Gooey GUI builder not available, will use command line arguments.\n'
                   'Install with "pip install Gooey". See README')

import random

def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        epilog='Run with no arguments to open dialog for server IP', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = server_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args



class track_server_process(mp.Process):
    def __init__(self, queue_from_server:mp.Queue, server_port_lock:mp.Lock(), server_socket:socket, client_addr, track_name=None, game_mode=GAME_MODE, car_name=CAR_NAME, ignore_off_track=DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK, timeout_s=CLIENT_TIMEOUT_SEC):
        super(track_server_process, self).__init__()
        self.server_queue=queue_from_server
        self.server_port_lock=server_port_lock
        self.server_socket=server_socket # used for initial communication to client who has not yet sent anything to us on the new port
        self.client_addr = client_addr
        self.track_name = track_name
        self.game_mode=game_mode
        self.car_list = []
            # = car(image_name=car_image_name, name=car_name)
        # self.car_model = CarModel(track=track_name, car_name=car_name, ignore_off_track=ignore_off_track)
        self.timeout_s=timeout_s

    def recv_msg(self):



    def run(self):
        logger.info("Starting track process track {} for client {}".format(self.track_name, self.client_addr))

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # make a new datagram socket
        if self.timeout_s>0: client_sock.settimeout(self.timeout_s)
        # find range of ports we can try to open for client to connect to
        local_port_number=bind_socket_to_range(CLIENT_PORT_RANGE, client_sock)
        gameAddr=client_sock.getsockname() # get the port info for our local port
        logger.info('found free local UDP port address {}, sending initial CarState to client at {}'.format(gameAddr, self.client_addr))

        msg=('track_port',local_port_number)
        send_message(lock=self.server_port_lock, socket=self.server_socket,client_addr=self.client_addr, msg=msg) # send client the port they should use for this track

        # now we start loop by waiting for first message, 'send_state"

        data = self.car_model.car_state
        p = pickle.dumps(data)  # tobi measured about 600 bytes, without car_name
        client_sock.sendto(p, self.client_addr)
        logger.info('starting control/state loop (waiting for initial client control from {})'.format(self.client_addr))
        lastT = timer()
        first_control_msg_received=False
        client_sock.settimeout(0) # nonblocking reads to get new commands and respond with current state
        last_message_time=timer()
        while True:
            try:
                data,lastClientAddr = client_sock.recvfrom(2048) # get control input TODO add timeout and better dead client handling
                (cmd, payload) = pickle.loads(data)
                if cmd=='command':
                    if payload.quit:
                        logger.info('quit recieved from {}, ending control loop'.format(lastClientAddr))
                        break
                    if payload.reset_car:
                        logger.info('reset recieved from {}, resetting car'.format(lastClientAddr))
                    command=payload
                    # self.car.car_state.update(self.car_model) # update user observed car state from model
                    # logger.info('sending car_state={}'.format(car.car_state))
                    payload=(dtSec, carThread.car.car_state) # send (dt,car_state) to client # todo instrument to see how big is data in bytes
                    message=('car_state',payload)
                    p=pickle.dumps(message) # about 840 bytes
                    client_sock.sendto(p,lastClientAddr)
                else:
                    logger.warning('unknowm cmd {} received; ignoring'.format(cmd))
                    continue

            except OSError as oserror:
                logger.warning('{}: garbled command or timeout, ending control loop thread'.format(oserror))
                break
            if not first_control_msg_received:
                logger.info('got first command from client at {}'.format(lastClientAddr))
                first_control_msg_received=True
            now=timer()
            dtSec = now-lastT # compute local real timestep, done on server to prevent accelerated real time cheating
            lastT=now
            self.car_model.update(dtSec=dtSec, command=command)
            self.track_name.car_completed_round(self.car_model)
            self.car.car_state=self.car_model.car_state
            sleep(1./FPS)

        logger.info('closing client socket, ending thread (server main thread waiting for new connection)')
        client_sock.close()

def send_message(socket:socket, lock:mp.Lock, client_addr, msg:object):
    try:
        lock.acquire()
        p=pickle.dumps(msg)
        socket.sendto(p, client_addr)
    finally:
        lock.release()


def make_track_process(track_name, client_addr) -> bool:
    if track_processes[track_name]!=None:
        return track_processes[track_name]
    logger.info('model server starting a new track_server_process for track {} for client at {}'.format(track_name, client_addr))
    q=mp.queues.Queue()
    track_quues[track_name]=q
    track_process = track_server_process(q, server_port_lock=server_port_lock, client_addr=client_addr, track_name=track_name, ignore_off_track=args.ignore_off_track, timeout_s=args.timeout_s)
    track_processes[track_name]=track_process
    track_processes[track_name].start()
    return track_processes[track_name]

def add_car_to_track(track_name, game_mode, car_name, client_addr):
    track_process=make_track_process(client_addr, track=track_name, car_name=car_name, ignore_off_track=DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK, timeout_s=CLIENT_TIMEOUT_SEC)
    q=track_quues[track_name]
    q.put(('add_car',game_mode, car_name, client_addr))


def add_spectator_to_track(track_name):
    track_process=make_track_process(track_name, client_addr)
    q=track_quues[track_name]
    q.put(('add_spectator',client_addr))

def stop_track_process(track_name):
    pass

def remove_car_from_track(track_name, car_name, client_addr):
    pass


if __name__ == '__main__':
    try:
        ga = Gooey(get_args, program_name="l2race server", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey"')
    args = get_args()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('', SERVER_PORT)) # bind to empty host, so we can receive from anyone on this port
    logger.info("waiting on {}".format(str(server_socket)))
    server_port_lock=mp.Lock() # proceeses get passed this lock to initiate connections using it (but only once, at start)

    # We fork an mp process for each track that might have one or more cars and spectators.

    # There is only a single instance of each track. Any clients that want to use that track share it.

    # Each track also gets a Queue which main process uses to tell it when there are new cars for it.

    # Each track runs single threaded to model all the cars in real time, and responds to commands (or spectate state requests) sent from clients with car states
    # of all the cars on that track.

    # Flow is like this:
    # 1. Server waits on SERVER_PORT
    # 2. Client sends newcar to SERVER_PORT
    # 3. Server responds to same port on client with ack and new port
    # 4. Client talks to track process on new port
    # That way, client initiates communication on new port and should be able to recieve on it

    track_names=list_tracks()
    emtpy_track_dict = {k:None for k in track_names} # each entry holds all the clients for each track name
    track_processes = emtpy_track_dict # each entry holds the track objects for each track name
    track_quues=emtpy_track_dict
    track_clients = emtpy_track_dict # each entry holds all the clients for each track name


    while True: # todo add KeyboardInterrupt exception handling, also SIG_TERM
        try:
            server_port_lock.acquire()
            data, client_addr = server_socket.recvfrom(1024)  # buffer size is 1024 bytes
        finally:
            server_port_lock.release()
        try:
            (cmd, payload) = pickle.loads(data)
        except pickle.UnpicklingError as ex:
            logger.warning('{}: garbled command, ignoring. Client should send 4-tuple (cmd, track_name, game_mode, car_name).\n '
                           'cmd="newcar"\n'
                           'track_name=<track_filename>\n'
                           'game_mode="solo|multi\n'
                           'car_name=<string_label_for_car>'.format(ex))
            continue

        logger.info('received cmd "{}" with payload "{}" from {}'.format(cmd, payload, client_addr))
        # todo handle multiple cars on one track, provide option for unique track for testing single car

        if cmd == 'newcar': # todo add arguments with newcar like driver/car name
            track_name, game_mode, car_name=payload
            add_car_to_track(track_name,game_mode,car_name,client_addr)
        elif cmd=='spectate':
            w='spectate not available yet'
            logger.warning(w)
            message=('warning',w)
            p=pickle.dumps(message) # about 840 bytes
            server_socket.sendto(p, client_addr)
        else:
            logger.warning('model server received unknown cmd={}'.format(cmd))

