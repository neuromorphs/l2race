# structure to hold driver control input
class car_command:
    """
    Car control commands from software agent or human driver, i.e., the throttle, steering, and brake input.

    Also includes reverse and autodrive_enabled boolean flags.
    """

    def __init__(self):
        # todo why are these floats and bools initialized to None?
        self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        self.brake=0  # bounded from 0-1
        self.reverse=False # boolean reverse gear
        self.autodrive_enabled = False # boolean activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def __str__(self):
        try:
            s = 'steering={:.2f}, throttle={:.2f}, brake={:.2f} reverse={} auto={}'.format(self.steering, self.throttle, self.brake, self.reverse, self.autodrive_enabled)
        except TypeError:
            s = 'car command contains None!'
        return s

    def complete_default(self):
        ''' todo what does this method do?'''
        if self.steering is None:
            self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        if self.throttle is None:
            self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        if self.brake is None:
            self.brake=0  # bounded from 0-1
        if self.reverse is None:
            self.reverse=False # boolean reverse gear
        if self.autodrive_enabled is None:
            self.autodrive_enabled = False # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def add_command(self, command):
        ''' todo what is this method for?'''
        if self.steering is None:
            self.steering=command.steering  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        if self.throttle is None:
            self.throttle=command.throttle  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        if self.brake is None:
            self.brake=command.brake  # bounded from 0-1
        if self.reverse is None:
            self.reverse=command.reverse # boolean reverse gear
        if self.autodrive_enabled is None:
            self.autodrive_enabled = command.autodrive_enabled # activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

