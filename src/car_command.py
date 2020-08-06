# structure to hold driver control input
class car_command:
    """ Car control commands from software agent or human driver
    """

    def __init__(self):
        self.steering=None  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        self.throttle=None  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        self.brake=None  # bounded from 0-1
        self.reverse=None # boolean reverse gear
        self.reset_car=None # in debugging mode, restarts car at starting line
        self.restart_client=None # abort current run (server went down?) and restart from scratch
        self.quit=None # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        self.auto = None # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def __str__(self):
        try:
            s = 'steering={:.2f}, throttle={:.2f}, brake={:.2f} reverse={}'.format(self.steering, self.throttle, self.brake,self.reverse)
        except TypeError:
            s = 'car command contains None!'
        return s

    def complete_default(self):
        if self.steering is None:
            self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        if self.throttle is None:
            self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        if self.brake is None:
            self.brake=0  # bounded from 0-1
        if self.reverse is None:
            self.reverse=False # boolean reverse gear
        if self.reset_car is None:
            self.reset_car=False # in debugging mode, restarts car at starting line
        if self.restart_client is None:
            self.restart_client=False # abort current run (server went down?) and restart from scratch
        if self.quit is None:
            self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        if self.auto is None:
            self.auto = False # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def add_command(self, command):
        if self.steering is None:
            self.steering=command.steering  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        if self.throttle is None:
            self.throttle=command.throttle  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        if self.brake is None:
            self.brake=command.brake  # bounded from 0-1
        if self.reverse is None:
            self.reverse=command.reverse # boolean reverse gear
        if self.reset_car is None:
            self.reset_car=command.reset_car # in debugging mode, restarts car at starting line
        if self.restart_client is None:
            self.restart_client=command.restart_client # abort current run (server went down?) and restart from scratch
        if self.quit is None:
            self.quit=command.quit # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        if self.auto is None:
            self.auto = command.auto # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

