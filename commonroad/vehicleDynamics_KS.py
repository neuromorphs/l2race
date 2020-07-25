from .longitudinalParameters import LongitudinalParameters
from .steeringConstraints import steeringConstraints
from .accelerationConstraints import accelerationConstraints
import numpy as np
from . import vehicleParameters
# from .vehicleParameters import VehicleParameters, vehicle_params_type

# import numba as nb
# from numba import float64 as f64 # use f64 to type explictly for list elements to tell numba it is just scalar value
# from numba import jit, deferred_type
#
# #https://stackoverflow.com/questions/53900084/problem-with-reflected-list-signature-in-numba
# fa=nb.types.List(nb.float64, reflected=False) # define numba type of list of float

# @jit(fa(fa, fa, vehicle_params_type), nopython=True)
def vehicleDynamics_KS(x,uInit,p):
    # vehicleDynamics_KS - kinematic single-track vehicle dynamics
    #
    # Syntax:  
    #    f = vehicleDynamics_KS(x,u,p)
    #
    # Inputs:
    #    x - vehicle state vector
    #    u - vehicle input vector
    #    p - vehicle parameter vector
    #
    # Outputs:
    #    f - right-hand side of differential equations
    #
    # Example: 
    #
    # Other m-files required: none
    # Subfunctions: none
    # MAT-files required: none
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      12-January-2017
    # Last update:16-December-2017
    # Last revision:---

    #------------- BEGIN CODE --------------

    #create equivalent kinematic single-track parameters
    l = (p.a + p.b)

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle

    #u1 = steering angle velocity of front wheels
    #u2 = longitudinal acceleration

    #consider steering constraints
    pl=p.longitudinal
    ps=p.steering
    x3=x[3]
    u1=uInit[1]
    a=accelerationConstraints(x3,u1,pl)
    s=steeringConstraints((x[2]),(uInit[0]),ps)
    u = [s,a]

    #system dynamics
    f = [(x[3])*np.cos((x[4])),
        (x[3])*np.sin((x[4])),
        (u[0]),
        (u[1]),
        (x[3])/l*np.tan((x[2]))] # every element is an f64
    
    return f

    #------------- END OF CODE --------------
