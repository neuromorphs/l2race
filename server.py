import argparse
import socket, pickle
import threading
import argcomplete
from timeit import default_timer as timer

from src.l2race_utils import bind_socket_to_range
from src.my_args import server_args
from src.car_model import CarModel
from src.car import car
from src.globals import *
from src.track import track
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



class ServerCarThread(threading.Thread): # change to multiprocessing, add queue to share data to other processes
    def __init__(self, addr, track, game_mode=GAME_MODE, car_name=CAR_NAME, ignore_off_track=DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK, timeout_s=CLIENT_TIMEOUT_SEC):
        threading.Thread.__init__(self)
        self.clientAddr = addr
        self.track = track
        self.game_mode=game_mode
        car_names = ['car_1', 'car_2']
        car_image_name = random.choice(car_names)
        self.car = car(image_name=car_image_name, name=car_name)
        self.car_model = CarModel(track=track, car_name=car_name, ignore_off_track=ignore_off_track)
        self.timeout_s=timeout_s


    def run(self):
        logger.info("Starting car thread for "+str(self.clientAddr))
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # make a new datagram socket
        if self.timeout_s>0: clientSock.settimeout(self.timeout_s)
        # find range of ports we can try to open for client to connect to
        bind_socket_to_range(CLIENT_PORT_RANGE, clientSock)
        gameAddr=clientSock.getsockname() # get the port info for our local port
        logger.info('found free local UDP port address {}, sending initial CarState to client at {}'.format(gameAddr,self.clientAddr))
        data = self.car_model.car_state
        p = pickle.dumps(data)  # tobi measured about 600 bytes, without car_name
        clientSock.sendto(p, self.clientAddr)
        logger.info('starting control/state loop (waiting for initial client control from {})'.format(self.clientAddr))
        lastT = timer()
        first_control_msg_received=False
        while True:
            try:
                data,lastClientAddr = clientSock.recvfrom(2048) # get control input TODO add timeout and better dead client handling
                command = pickle.loads(data)
                if command.quit:
                    logger.info('quit recieved from {}, ending control loop'.format(lastClientAddr))
                    break
                if command.reset_car:
                    logger.info('reset recieved from {}, resetting car'.format(lastClientAddr))
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
            self.track.car_completed_round(self.car_model)
            self.car.car_state=self.car_model.car_state
            # self.car.car_state.update(self.car_model) # update user observed car state from model
            # logger.info('sending car_state={}'.format(car.car_state))
            data=(dtSec, carThread.car.car_state) # send (dt,car_state) to client # todo instrument to see how big is data in bytes
            p=pickle.dumps(data) # about 840 bytes
            clientSock.sendto(p,lastClientAddr)

        logger.info('closing client socket, ending thread (server main thread waiting for new connection)')
        clientSock.close()


if __name__ == '__main__':
    try:
        ga = Gooey(get_args, program_name="l2race server", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey"')
    args = get_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', SERVER_PORT)) # bind to empty host, so we can receive from anyone on this port
    logger.info("waiting on {}".format(str(sock)))
    clients = dict()

    while True: # todo add KeyboardInterrupt exception handling, also SIG_TERM
        data, clientAddr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        try:
            (cmd, payload) = pickle.loads(data)
        except pickle.UnpicklingError as ex:
            logger.warning('{}: garbled command, ignoring. Client should send 4-tuple (cmd, track_name, game_mode, car_name).\n '
                           'cmd="newcar"\n'
                           'track_name=<track_filename>\n'
                           'game_mode="solo|multi\n'
                           'car_name=<string_label_for_car>'.format(ex))
            continue

        logger.info('received cmd "{}" with payload "{}" from {}'.format(cmd, payload, clientAddr))
        # todo handle multiple cars on one track, provide option for unique track for testing single car

        if cmd == 'newcar': # todo add arguments with newcar like driver/car name
            track_name, game_mode, car_name=payload
            current_track = track(track_name=track_name) # todo reuse track for multi mode
            logger.info('model server starting a new ServerCarThread car named {} on track {} in game_mode {} for client at {}'.format(car_name, track_name, game_mode, clientAddr))
            carThread = ServerCarThread(addr=clientAddr, track=current_track, game_mode=game_mode, car_name=car_name, ignore_off_track=args.ignore_off_track, timeout_s=args.timeout_s)
            clients[clientAddr] = carThread
            carThread.start()
        else:
            logger.warning('model server received unknown cmd={}'.format(cmd))

