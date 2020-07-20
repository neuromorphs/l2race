# structure to hold driver control input
class car_input:
    """ Car control input from software agent or human driver
    """

    def __init__(self):
        self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        self.brake=0  # bounded from 0-1
        self.reverse=False # boolean reverse gear
        self.reset=False # in debugging mode, restarts car at starting line
        self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller

    def __str__(self):
        return 'steering={:.2f}, throttle={:.2f}, brake={:.2f}'.format(self.steering, self.throttle, self.brake)
