# arguments for l2race client and server
import os
from src.globals import *
import logging

logger = logging.getLogger(__name__)

def client_args(parser):
    # check and add prefix if running script in subfolder

    # general arguments for output folder, overwriting, etc
    clientGroup = parser.add_argument_group('Client arguments:')
    clientGroup.add_argument("--host", type=str, default=SERVER_HOST, help="Sost IP of model server.")
    clientGroup.add_argument("--port", type=int, default=SERVER_PORT, help="Server port address for initiating connections.")
    clientGroup.add_argument("--fps", type=int, default=FPS, help="Frame rate on client side (server always sets time to real time")
    clientGroup.add_argument("--joystick", type=int, default=JOYSTICK_NUMBER, help="Desired joystick number")

    return parser

def server_args(parser):
    # check and add prefix if running script in subfolder

    serverGroup = parser.add_argument_group('Server arguments:')
    serverGroup.add_argument(
        "--ignore_off_track", action='store_true',
        help="ignore when car goes off track (for testing car dynamics more easily)")

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


