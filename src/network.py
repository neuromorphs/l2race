import socket

from src.mylogger import mylogger
logger=mylogger(__name__)
HOST = 'localhost'


def getudpsocket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, SERVER_PORT))
    logger.info("waiting on {}".format(str(s)))
    return s