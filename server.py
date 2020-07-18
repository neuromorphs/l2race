import socket, pickle
import threading

from car_model import CarModel
from src.car import Car
from src.mylogger import mylogger
from src.globals import *
import pygame

from track import Track

logger=mylogger(__name__)

class ServerCarThread(threading.Thread):
    def __init__(self, addr, track):
        threading.Thread.__init__(self)
        self.clientAddr=addr
        self.track=track
        self.car = Car(track=track)
        self.car_model=CarModel(track=track)

    def run(self):
        clock = pygame.time.Clock()
        logger.info("Starting car thread for "+str(self.clientAddr))
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        clientSock.bind(("", 0))
        gameAddr=clientSock.getsockname()
        logger.info('found free UDP port address {}, sending initial CarState to client'.format(gameAddr))
        p = pickle.dumps(self.car_model.car_state)
        clientSock.sendto(p,self.clientAddr)
        logger.info('starting control/state loop')
        lastT=clock.get_time()
        while True:
            # todo update time here according to real time,
            #  i.e. base timestamp on real time since last message from client
            data,clientAddr = clientSock.recvfrom(4096) # todo add timeout to handle dropped packets

            car_input = pickle.loads(data)
            if car_input.quit:
                logger.info('quit recieved from {}, ending control loop'.format(self.clientAddr))
                break
            if car_input.reset:
                logger.info('reset recieved from {}, resetting car'.format(self.clientAddr))
            dt = .1 # todo debug  #clock.tick()/1000.0
            self.car_model.update(dt=dt,input=car_input)
            self.car.car_state=self.car_model.car_state
            # self.car.car_state.update(self.car_model) # update user observed car state from model
            # logger.info('sending car_state={}'.format(car.car_state))
            data=(dt, carThread.car.car_state) # send (dt,car_state) to client # todo instrument to see how big is data in bytes
            p=pickle.dumps(data)
            clientSock.sendto(p,clientAddr)

        logger.info('closing client socket')
        clientSock.close()


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("localhost", SERVER_PORT))
    logger.info("waiting on {}".format(str(sock)))
    clients = dict()
    track=Track()

    while True:
        data, clientAddr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        cmd=pickle.loads(data)
        logger.info('received message: {} from {}'.format(cmd, clientAddr))

        if cmd=='newcar':
            logger.info('starting a new ServerCarThread for client at {}'.format(clientAddr))
            carThread=ServerCarThread(clientAddr, track)
            clients[clientAddr]=carThread
            carThread.start()

