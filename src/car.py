
from pygame.math import Vector2
from math import sin, cos, radians, degrees, copysign
from src.mylogger import mylogger
logger = mylogger(__name__)
from src.car_state import car_state
from src.game import WIDTH, HEIGHT, PPU

class Car:
    """
    Model of car including dynamical model
    """


    def __init__(self, x, y, angle=0.0, length=4, max_steering=50, max_acceleration=5.0):
        self.car_state = car_state(x,y)

    def update(self, dt, input):
        
        acceleration=input.throttle*self.car_state.max_acceleration-input.brake * self.car_state.brake_deceleration
        self.car_state.speed =self.car_state.velocity.length()  # in case something external touched component of velocity, like collision

        if acceleration==0:
            acceleration=-self.car_state.free_deceleration

        dacc=acceleration * dt
        self.car_state.speed += dacc
        if self.car_state.speed<0:
            self.car_state.speed=0
        elif self.car_state.speed>self.car_state.max_speed:
            self.car_state.speed=self.car_state.max_speed

        steer=-input.steering*self.car_state.max_steering
        if not steer==0:
            turning_radius = self.car_state.length / sin(radians(steer))
            angular_velocity = self.car_state.speed / turning_radius
        else:
            angular_velocity=0
        self.car_state.angle_deg += degrees(angular_velocity) * dt

        self.car_state.velocity=Vector2(self.car_state.speed * cos(radians(self.car_state.angle_deg)), -self.car_state.speed * sin(radians(self.car_state.angle_deg)))

        self.car_state.position += self.car_state.velocity * dt

        w = WIDTH / PPU
        h = HEIGHT / PPU
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

    def reset(self):
        self.car_state.position = Vector2(5, 5)