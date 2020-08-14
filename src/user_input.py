# user input, like quit, show statistics, etc.
# Separate from car_command which is just for controlling car

class user_input():
    def __init__(self):
        self.restart_car=False # in debugging mode, restarts car at starting line
        self.restart_client=False # abort current run (server went down?) and restart from scratch
        self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
