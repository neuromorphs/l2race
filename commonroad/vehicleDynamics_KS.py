from .steeringConstraints import steeringConstraints
from .accelerationConstraints import accelerationConstraints
import math

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
    l = p.a + p.b 

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle

    #u1 = steering angle velocity of front wheels
    #u2 = longitudinal acceleration
    
    #consider steering constraints
    u = [];
    u.append(steeringConstraints(x[2],uInit[0],p.steering)) # different name uInit/u due to side effects of u
    #consider acceleration constraints
    u.append(accelerationConstraints(x[3],uInit[1],p.longitudinal)) # different name uInit/u due to side effects of u

    #system dynamics
    f = [x[3]*math.cos(x[4]), 
        x[3]*math.sin(x[4]), 
        u[0], 
        u[1], 
        x[3]/l*math.tan(x[2])]
    
    return f

    #------------- END OF CODE --------------
