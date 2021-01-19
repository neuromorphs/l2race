# from .vehicleParameters import steering_type
import numba as nb
from numba import jit
#https://stackoverflow.com/questions/53900084/problem-with-reflected-list-signature-in-numba
f=nb.float64

# @jit(f(f, f, steering_type),nopython=True)
cpdef double steeringConstraints(double steeringAngle,double steeringVelocity,object p):
    # steeringConstraints - adjusts the steering velocity based on steering
    # constraints
    #
    # Syntax:  
    #    steeringConstraints(steeringAngle,steeringVelocity,p)
    #
    # Inputs:
    #    steeringAngle - steering angle
    #    steeringVelocity - steering velocity
    #    p - steering parameter structure
    #
    # Outputs:
    #    steeringVelocity - steering velocity
    #
    # Example: 
    #
    # Other m-files required: none
    # Subfunctions: none
    # MAT-files required: none
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      15-December-2017
    # Last update:  ---
    # Last revision:---

    #------------- BEGIN CODE --------------

    #steering limit reached?
    if (steeringAngle<=p.min and steeringVelocity<=0) or (steeringAngle>=p.max and steeringVelocity>=0):
        steeringVelocity = 0
    elif steeringVelocity<=p.v_min:
        steeringVelocity = p.v_min
    elif steeringVelocity>=p.v_max:
        steeringVelocity = p.v_max 
        
    return steeringVelocity

    #------------- END OF CODE --------------
