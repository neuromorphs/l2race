def init_KS(initState):
    # init_KS - generates the initial state vector for the kinematic single-track model

    # Syntax:  
    #     x0 = init_KS(initState, p)
    #
    # Inputs:
    #     initState - core initial states
    #
    # Outputs:
    #     x0 - initial state vector
    #
    # Example: 
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      11-January-2017
    # Last update:  16-December-2017
    # Last revision:---


    #------------- BEGIN CODE --------------

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle

    #u1 = steering angle velocity of front wheels
    #u2 = ongitudinal acceleration

    #obtain initial states from vector
    sx0 = initState[0] 
    sy0 = initState[1] 
    delta0 = initState[2]
    vel0 = initState[3] 
    Psi0 = initState[4] 

    #sprung mass states
    x0 = [] # init initial state vector
    x0.append(sx0)  # x-position in a global coordinate system
    x0.append(sy0)  # y-position in a global coordinate system
    x0.append(delta0)  # steering angle of front wheels
    x0.append(vel0)  # velocity
    x0.append(Psi0)  # yaw angle

    return x0

    #------------- END OF CODE --------------
