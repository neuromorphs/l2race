import socket, pickle
import threading

from src.car import Car
from src.mylogger import mylogger
SERVER_PORT = 50000

logger=mylogger(__name__)

class ServerCarThread(threading.Thread):
    def __init__(self, addr):
        threading.Thread.__init__(self)
        self.clientAddr=addr

    def run(self):
        logger.info("Starting car thread for "+str(self.clientAddr))
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        clientSock.bind(("", 0))
        gameAddr=clientSock.getsockname()
        logger.info('found free UDP port address {}'.format(gameAddr))
        p = pickle.dumps(gameAddr)
        clientSock.sendto(p,self.clientAddr)
        logger.info('starting control/state loop')
        car = Car(0, 0)
        while True:
            data,clientAddr = clientSock.recvfrom(4096)
            (dt,car_input) = pickle.loads(data)
            # logger.info('got dt={:1f}ms, car_input={}'.format(dt*1e3, car_input))
            car.update(dt,car_input)
            # logger.info('sending car_state={}'.format(car.car_state))
            p=pickle.dumps(car.car_state)
            clientSock.sendto(p,clientAddr)

        clientSock.close()

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("localhost", SERVER_PORT))
    logger.info("waiting on {}".format(str(sock)))
    clients = dict()

    while True:
        data, clientAddr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        cmd=pickle.loads(data)
        logger.info('received message: {} from {}'.format(cmd, clientAddr))

        if cmd=='newcar':
            logger.info('starting a new ServerCarThread for client at {}'.format(clientAddr))
            car=ServerCarThread(clientAddr)
            clients[clientAddr]=car
            car.start()

