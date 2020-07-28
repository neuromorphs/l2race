# structure to hold driver control input
class car_command:
    """ Car control commands from software agent or human driver
    """

    def __init__(self):
        self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        self.brake=0  # bounded from 0-1
        self.reverse=False # boolean reverse gear
        self.reset_car=False # in debugging mode, restarts car at starting line
        self.restart_client=False # abort current run (server went down?) and restart from scratch
        self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        self.auto = False # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def __str__(self):
        return 'steering={:.2f}, throttle={:.2f}, brake={:.2f} reverse={}'.format(self.steering, self.throttle, self.brake,self.reverse)
