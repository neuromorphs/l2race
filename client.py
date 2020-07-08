"""
Client racecar agent

"""
from src.game import Game
import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import logging
import socket, pickle

from src.mylogger import mylogger

logger=mylogger(__name__)

if __name__ == '__main__':
    game = Game()
    game.run()