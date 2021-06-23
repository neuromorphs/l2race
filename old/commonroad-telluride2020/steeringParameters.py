from numba import float64  #
from numba.experimental import jitclass

spec = [
    ('min', float64),
    ('max', float64),
    ('v_min', float64),
    ('v_max', float64),
]
# @jitclass(spec)
class SteeringParameters():
    def __init__(self):
        #constraints regarding steering
        self.min = 0.  #minimum steering angle [rad]
        self.max = 0. #maximum steering angle [rad]
        self.v_min = 0.  #minimum steering velocity [rad/s]
        self.v_max = 0.  #maximum steering velocity [rad/s]
