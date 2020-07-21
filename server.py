import argparse
import socket, pickle
import threading
import argcomplete
from timeit import default_timer as timer

from src.args import server_args
from src.car_model import CarModel
from src.car import Car
from src.globals import *
from src.track import Track
# may only apply to windows
from src.mylogger import mylogger
logger=mylogger(__name__)
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
    parser = server_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args



class ServerCarThread(threading.Thread):
    def __init__(self, addr, track):
        threading.Thread.__init__(self)
        self.clientAddr=addr
        self.track=track
        self.car = Car(track=track)
        self.car_model=CarModel(track=track)

    def run(self):
        logger.info("Starting car thread for "+str(self.clientAddr))
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # make a new datagram socket
        clientSock.bind(('', 0)) # bind to port 0 to get a random free port
        gameAddr=clientSock.getsockname() # get the port info for our local port
        logger.info('found free local UDP port address {}, sending initial CarState to client at {}'.format(gameAddr,self.clientAddr))
        p = pickle.dumps(self.car_model.car_state) # tobi measured about 600 bytes
        clientSock.sendto(p,self.clientAddr)
        logger.info('starting control/state loop (waiting for initial client control')
        lastT=timer()
        while True:
            data,clientAddr = clientSock.recvfrom(4096) # get control input
            command = pickle.loads(data)
            if command.quit:
                logger.info('quit recieved from {}, ending control loop'.format(self.clientAddr))
                break
            if command.reset:
                logger.info('reset recieved from {}, resetting car'.format(self.clientAddr))
            now=timer()
            dtSec = now-lastT # compute local real timestep, done on server to prevent accelerated real time cheating
            lastT=now
            self.car_model.update(dtSec=dtSec, command=command)
            self.car.car_state=self.car_model.car_state
            # self.car.car_state.update(self.car_model) # update user observed car state from model
            # logger.info('sending car_state={}'.format(car.car_state))
            data=(dtSec, carThread.car.car_state) # send (dt,car_state) to client # todo instrument to see how big is data in bytes
            p=pickle.dumps(data) # about 750 bytes
            clientSock.sendto(p,clientAddr)

        logger.info('closing client socket')
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

    DO_NOT_RESET_CAR_WHEN_IT_GOES_OFF_TRACK=args.ignore_off_track

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', SERVER_PORT)) # bind to empty host, so we can receive from anyone on this port
    logger.info("waiting on {}".format(str(sock)))
    clients = dict()
    track=Track()

    while True:
        data, clientAddr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        cmd=pickle.loads(data)
        logger.info('received message: "{}" from {}'.format(cmd, clientAddr))

        if cmd=='newcar': # todo add arguments with newcar like driver/car name
            logger.info('model server starting a new ServerCarThread for client at {}'.format(clientAddr))
            carThread=ServerCarThread(clientAddr, track)
            clients[clientAddr]=carThread
            carThread.start()
        else:
            logger.warning('model server received unknown cmd={}'.format(cmd))

