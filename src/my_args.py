# arguments for l2race client and server
import os

import src.car_model
from src.globals import *
import logging

logger = logging.getLogger(__name__)

def client_args(parser):
    # check and add prefix if running script in subfolder

    # general arguments for output folder, overwriting, etc
    clientGroup = parser.add_argument_group('Client arguments:')
    clientGroup.add_argument("--host", type=str, default=SERVER_HOST, help="Sost IP of model server.")
    clientGroup.add_argument("--port", type=int, default=SERVER_PORT, help="Server port address for initiating connections.")
    clientGroup.add_argument("--fps", type=int, default=FPS, help="Frame rate on client side (server always sets time to real time).")
    clientGroup.add_argument("--joystick", type=int, default=JOYSTICK_NUMBER, help="Desired joystick number, starting with 0.")
    clientGroup.add_argument("--timeout_s", type=float, default=SERVER_TIMEOUT_SEC, help="Socket timeout in seconds for communication with model server.")
    clientGroup.add_argument("--record", action='store_true', help="record data to date-stamped filename, e.g. --record will write datestamped files named '{}-XXX.csv' in folder '{}, where XXX is a date/timestamp'.".format(DATA_FILENAME_BASE, DATA_FOLDER_NAME))

    return parser

def server_args(parser):
    # check and add prefix if running script in subfolder

    serverGroup = parser.add_argument_group('Server arguments:')
    serverGroup.add_argument("--ignore_off_track", action='store_true', help="ignore when car goes off track (for testing car dynamics more easily)")
    serverGroup.add_argument("--timeout_s", type=int, default=CLIENT_TIMEOUT_SEC, help="server timeout in seconds before it ends thread for handling a car model")
    # serverGroup.add_argument("--model", type=str, default=src.car_model.MODEL, help="server timeout in seconds before it ends thread for handling a car model")

    return parser

def write_args_info(args, path)-> str:
    '''
    Writes arguments to logger and file named from startup __main__
    Parameters
    ----------
    args: parser.parse_args()

    Returns
    -------
    full path to file
    '''
    import __main__
    arguments_list = 'arguments:\n'
    for arg, value in args._get_kwargs():
        arguments_list += "{}:\t{}\n".format(arg, value)
    logger.info(arguments_list)
    basename = os.path.basename(__main__.__file__)
    argsFilename = basename.strip('.py') + '-args.txt'
    filepath = os.path.join(path, argsFilename)
    with open(filepath, "w") as f:
        f.write(arguments_list)
    return filepath


