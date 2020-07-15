
from pygame.math import Vector2
from math import sin, cos, radians, degrees, copysign
from src.mylogger import mylogger
from src.track import Track

logger = mylogger(__name__)
from src.carstate import CarState
from src.globals import SCREEN_WIDTH, SCREEN_HEIGHT, PPU

class Car:
    """
    Model of car including dynamical model. Based on https://asmedigitalcollection.asme.org/dynamicsystems/article/142/2/021004/1066044/Toward-Automated-Vehicle-Control-Beyond-the

    .. figure:: ../media/coordinates.png
    """


    def __init__(self, x=0, y=0):
        self.car_state = CarState(x, y)
        self.track=Track() # TODO for now just use default track
        self.closestTrackVertex=None

    def update(self, dt, input):

        if input.reset:
            logger.info('resetting car')
            self.reset()

        self.car_state.speed =self.car_state.velocity.length()  # in case something external touched component of velocity, like collision
        acceleration=input.throttle*self.car_state.max_acceleration-input.brake * self.car_state.brake_deceleration
        if input.reverse: # todo fix, logic incorrect
            acceleration=-acceleration

        if acceleration==0:
            acceleration=-self.car_state.free_deceleration

        dacc=acceleration * dt
        self.car_state.speed += dacc
        if self.car_state.speed<0:
            self.car_state.speed=0
        elif self.car_state.speed>self.car_state.max_speed:
            self.car_state.speed=self.car_state.max_speed

        self.car_state.steering=-input.steering*(self.car_state.max_steering*((2*self.car_state.max_speed-self.car_state.speed)/(2*self.car_state.max_speed)))
        if not self.car_state.steering==0:
            turning_radius = self.car_state.length / sin(radians(self.car_state.steering))
            angular_velocity = self.car_state.speed / turning_radius
        else:
            angular_velocity=0
        self.car_state.angle_deg += degrees(angular_velocity) * dt

        self.car_state.velocity=Vector2(self.car_state.speed * cos(radians(self.car_state.angle_deg)), -self.car_state.speed * sin(radians(self.car_state.angle_deg)))

        self.car_state.position += self.car_state.velocity * dt

        self.locate()

        w = SCREEN_WIDTH
        h = SCREEN_HEIGHT
        if self.car_state.position.x > w:
            self.car_state.position.x = w
            self.car_state.velocity.x = -self.car_state.velocity.x
        elif self.car_state.position.x < 0:
            self.car_state.position.x = 0
            self.car_state.velocity.x = -self.car_state.velocity.x
        if self.car_state.position.y > h:
            self.car_state.position.y = h
            self.car_state.velocity.y = -self.car_state.velocity.y
        elif self.car_state.position.y < 0:
            self.car_state.position.y = 0
            self.car_state.velocity.y = -self.car_state.velocity.y
        # logger.info(self.car_state)


    def locate(self):
        """ locates car on track and updates in the car_state"""
        # vs=self.track.vertices
        # minDist=None if self.closestTrackVertex==None else minDist=(self.closestTrackVertex-Vector2(vs[]))
        # for p in :

        pass


    def reset(self):
        """ reset car to starting position"""
        x=0
        y=0
        if self.track:
            x=self.track.vertices[0][0]
            y=self.track.vertices[0][0]
        self.car_state.position = Vector2(x,y) # todo reset to start line of track