# arguments for l2race client and server
import os

from l2race_settings import *
import logging

from track import list_tracks

logger = logging.getLogger(__name__)

def client_args(parser):
    # check and add prefix if running script in subfolder

    # general arguments for output folder, overwriting, etc
    clientServerGroup = parser.add_argument_group('Model server connection options:')
    clientServerGroup.add_argument("--host", type=str, default=SERVER_HOST, help="IP address or DNS name of model server.")
    clientServerGroup.add_argument("--port", type=int, default=SERVER_PORT, help="Server port address for initiating connections.")
    clientServerGroup.add_argument("--timeout_s", type=float, default=SERVER_TIMEOUT_SEC, help="Socket timeout in seconds for communication with model server.")

    clientInterfaceGroup = parser.add_argument_group('Interface arguments:')
    clientInterfaceGroup.add_argument("--fps", type=int, default=FPS, help="Frame rate on client side (server always sets time to real time).")
    clientInterfaceGroup.add_argument("--joystick", type=int, default=JOYSTICK_NUMBER, help="Desired joystick number, starting with 0.")

    clientOutputGroup = parser.add_argument_group('Output/Replay options:')
    clientOutputGroup.add_argument("--record", nargs='?',const='',  type=str, help="Record data to date-stamped filename with optional <note>, e.g. --record will write datestamped files named '{}-<track_name>-<car_name>-<note>-TTT.csv' in folder '{}, where note is optional note and TTT is a date/timestamp\'.".format(DATA_FILENAME_BASE, DATA_FOLDER_NAME))
    clientOutputGroup.add_argument("--replay", nargs='?', const='last', type=str, help="Replay one or more CSV recordings. If 'last' or no file is supplied, play the most recent recording in the '{}' folder.".format(DATA_FOLDER_NAME))

    clientControllerGroup = parser.add_argument_group('Control/Modeling arguments:')
    clientControllerGroup.add_argument("--autodrive",type=str,nargs=2, default=[AUTODRIVE_MODULE,AUTODRIVE_CLASS],help="The autodrive module and class to be run when autodrive is enabled on controller. Pass it the module (i.e. folder.file without .py) and the class within the file.")
    clientControllerGroup.add_argument("--carmodel", type=str, nargs=2, default=[CAR_MODEL_MODULE, CAR_MODEL_CLASS], help="Your client car module and class and class to be run as ghost car when model evaluation is enabled on controller. Pass it the module (i.e. folder.file without .py) and the class within the file.")

    clientSensorGroup = parser.add_argument_group('Sensor arguments:')
    clientSensorGroup.add_argument("--lidar", type=float, nargs='?', default=None, const=5.0,
                                   help="Draw the point at which car would hit the track edge if moving on a straight line. "
                                        "The numerical value gives precision in pixels with which this point is found.")

    clientTrackCarMode = parser.add_argument_group('Track car/spectate options:')

    clientTrackCarMode.add_argument("--track_name", type=str, default=TRACK_NAME, choices=list_tracks(), help="Name of track. Available tracks are in the '{}' folder, defined by globals.TRACKS_FOLDER.".format(TRACKS_FOLDER))
    clientTrackCarMode.add_argument("--car_name", type=str, default=None, help="Name of this car (last 2 letters are randomly chosen each time).")
    clientTrackCarMode.add_argument("--spectate", action='store_true', help="Just be a spectator on the cars on the track.")

    # other options
    parser.add_argument('--log',type=str,default=str(logging.getLevelName(LOGGING_LEVEL)),help='Set logging level. From most to least verbose, choices are "DEBUG", "INFO", "WARNING".')
    return parser

def server_args(parser):
    # check and add prefix if running script in subfolder

    serverGroup = parser.add_argument_group('Server arguments:')
    serverGroup.add_argument("--allow_off_track", action='store_true', help="ignore when car goes off track (for testing car dynamics more easily)")
    serverGroup.add_argument('--log',type=str,default=str(logging.getLevelName(LOGGING_LEVEL)),help='Set logging level. From most to least verbose, choices are "DEBUG", "INFO", "WARNING".')
    serverGroup.add_argument("--port", type=int, default=SERVER_PORT, help="Server port address for initiating connections from clients.")
    # serverGroup.add_argument("--timeout_s", type=int, default=CLIENT_TIMEOUT_SEC, help="server timeout in seconds before it ends thread for handling a car model")
    # serverGroup.add_argument("--model", type=str, default=car_model.MODEL, help="server timeout in seconds before it ends thread for handling a car model")

    return parser

def write_args_info(args, filepath)-> str:
    """
    Writes arguments to logger and file named from startup __main__
    Parameters

    :param args: parser.parse_args()
    :param filepath: full path to logger output file

    :returns: full path to file
    """
    import __main__
    arguments_list = 'arguments:\n'
    for arg, value in args._get_kwargs():
        arguments_list += "{}:\t{}\n".format(arg, value)
    logger.info(arguments_list)
    with open(filepath, "w") as f:
        f.write(arguments_list)
    return filepath


