def steeringConstraints(steeringAngle,steeringVelocity,p):
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
