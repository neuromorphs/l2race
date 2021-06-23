from numba.experimental import jitclass
from numba import float64  #

spec = [
    ('v_min', float64),
    ('v_max', float64),
    ('v_switch', float64),
    ('a_max', float64),
]


# @jitclass(spec)
class LongitudinalParameters():
    def __init__(self):
        #constraints regarding longitudinal dynamics
        self.v_min = 0.  #minimum velocity [m/s]
        self.v_max = 0.   #minimum velocity [m/s]
        self.v_switch = 0.  #switching velocity [m/s]
        self.a_max = 0.  #maximum absolute acceleration [m/s^2]

