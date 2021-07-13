# user input, like quit, show statistics, etc.
# Separate from car_command which is just for controlling car

class user_input():
    """
    User input to l2race client.

    Includes the restart_car, restart_client, record_data, run_client_model, and quit commands
    """
    def __init__(self):
        self.restart_car=False # in debugging mode, restarts car at starting line
        self.restart_client=False # abort current run (server went down?) and restart from scratch
        self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        self.run_client_model=False # run the client model of car
        self.toggle_recording=False # record data
        self.open_playback_recording=False
        self.close_playback_recording=False
