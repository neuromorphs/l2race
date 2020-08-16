# utility methods
import logging
import os
import socket, pickle
from time import sleep
from timeit import default_timer as timer
from time import sleep as sleep


from src.globals import CLIENT_PORT_RANGE

# customized logger with color output
import logging

from src.globals import LOGGING_LEVEL
import numpy as np
from collections import deque
class circular_buffer(deque):
    def __init__(self, size=0):
        super(circular_buffer, self).__init__(maxlen=size)

    @property
    def average(self):  # TODO: Make type check for integer or floats
        return sum(self)/len(self)

    def hist(self):
        return np.histogram(np.array(self))


class loop_timer():
    """ simple game loop timer that sleeps for leftover time (if any) at end of each iteration"""
    LOG_INTERVAL_SEC=10
    NUM_SAMPLES=1000
    def __init__(self, rate_hz:float):
        ''' :param rate_hz: the target loop rate'''
        self.rate_hz=rate_hz
        self.start_loop()
        self.loop_counter=0
        self.last_log_time=0
        self.circ_buffer=circular_buffer(self.NUM_SAMPLES)
        self.first_call_done=False

    def start_loop(self):
        """ can be called to initialize the timer"""
        self.last_iteration_start_time=timer()

    def sleep_leftover_time(self):
        """ call at start or end of each iteration """
        now=timer()
        if not self.first_call_done:
            self.first_call_done=True
            return # don't sleep on first call at start of loop

        max_sleep=1./self.rate_hz
        dt=(now-self.last_iteration_start_time)
        leftover_time=max_sleep-dt
        self.circ_buffer.append(dt)
        if leftover_time>0:
            sleep(leftover_time)
        self.start_loop()
        self.loop_counter+=1
        if now-self.last_log_time>self.LOG_INTERVAL_SEC:
            self.last_log_time=now
            if leftover_time>0:
                logger.info('loop_timer slept for {:.1f}ms leftover time for desired loop interval {:.1f}ms'.format(leftover_time*1000,max_sleep*1000))
            else:
                logger.warning('loop_timer cannot achieve desired rate {}Hz, time ran over by {}ms compared with allowed time {}ms'.format(self.rate_hz, -leftover_time*1000, max_sleep*1000))
            logger.info('histogram of intervals (counts and bin edges in s)\n{}'.format(self.circ_buffer.hist()))

# https://stackoverflow.com/questions/1423345/can-i-run-a-python-script-as-a-service
import os, sys

def become_daemon(our_home_dir='.', out_log='/dev/null', err_log='/dev/null', pidfile='daemon.pid'):
    """ Make the current process a daemon.  """

    try:
        # First fork
        try:
            if os.fork() > 0:
                sys.exit(0)
        except OSError as e:
            logger.warning('Could not become daemon: fork #1 failed" (%d) %s\n' % (e.errno, e.strerror))
            # sys.exit(1)

        os.setsid()
        os.chdir(our_home_dir)
        os.umask(0)

        # Second fork
        try:
            pid = os.fork()
            if pid > 0:
                # You must write the pid file here.  After the exit()
                # the pid variable is gone.
                fpid = open(pidfile, 'wb')
                fpid.write(str(pid))
                fpid.close()
                sys.exit(0)
        except OSError as e:
            logger.critical('Could not become daemon: fork #2 failed" (%d) %s\n' % (e.errno, e.strerror))
            # sys.exit(1)

        si = open('/dev/null', 'r')
        so = open(out_log, 'a+', 0)
        se = open(err_log, 'a+', 0)
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
    except Exception as e:
        logger.warning('Could not become daemon (might only work under linux): {}'.format(e))

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def my_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger

logger = my_logger(__name__)

def set_logging_level(args): # todo still does not correctly affect all of our existing loggers.... tricky
    global LOGGING_LEVEL
    if args.log=='INFO':
        LOGGING_LEVEL=logging.INFO
    elif args.log=='DEBUG':
        LOGGING_LEVEL=logging.DEBUG
    elif args.log=='WARNING':
        LOGGING_LEVEL=logging.WARNING
    elif args.log=='CRITICAL':
        LOGGING_LEVEL=logging.CRITICAL
    else:
        logger.warning('unknown logging level {} specified, using default level {}'.format(args.logging_level,logger.getEffectiveLevel()))

def random_port_permutation(portrange):
    ''' find a free server port in range
    :arg portrange a string e.g. 10001-10010
    :arg clientSockthe local socket we try to bind to a local port number
    :returns the port number
    :raises RuntimeError if cannot find a free port in range
    '''

    s = portrange.split('-')
    if len(s) != 2:
        raise RuntimeError(
            'client port range {} should be of form start-end, e.g. 50100-50200'.format(portrange))
    start_port = int(s[0])
    end_port = int(s[1])
    r = np.random.permutation(np.arange(start_port, end_port))
    return r

def find_unbound_port_in_range(portrange:str):
    r=random_port_permutation(portrange)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # make a new datagram socket
    isbound = False
    for p in r:
        try:
            sock.bind(('0.0.0.0', p))  # bind to port 0 to get a random free port
            logger.info('bound socket {} to local port {}'.format(sock, p))
            isbound = True
            return p
        except:
            logger.warning('tried but could not bind to port {}'.format(p))
    if not isbound:
        raise RuntimeError('could not bind socket {} to any local port in range {}'.format(client_sock, portrange))


def checkAddSuffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path
    else:
        return os.path.splitext(path)[0]+suffix



# good codec, basically mp4 with simplest compression, packed in AVI,
# only 15kB for a few seconds
OUTPUT_VIDEO_CODEC_FOURCC = 'XVID'

def video_writer(output_path, height, width, frame_rate=30):
    """ Return a video writer.

    Parameters
    ----------
    output_path: str,
        path to store output video.
    height: int,
        height of a frame.
    width: int,
        width of a frame.
    frame_rate: int
        playback frame rate in Hz

    Returns
    -------
    an instance of cv2.VideoWriter.
    """

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC_FOURCC)
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        frame_rate,
        (width, height))
    logger.debug(
        'opened {} with {} https://www.fourcc.org/ codec, {}fps, '
        'and ({}x{}) size'.format(
            output_path, OUTPUT_VIDEO_CODEC_FOURCC, frame_rate,
            width, height))
    return out

def get_local_ip_address(): # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def open_ports():
    import upnpy
    logger.info('Attempting to open necessary UDP ports with upnpy version {} (https://github.com/5kyc0d3r/upnpy, https://upnpy.readthedocs.io/en/latest/)'.format(upnpy.__version__))
    logger.setLevel(logging.DEBUG)
    upnp = upnpy.UPnP()

    # Discover UPnP devices on the network
    # Returns a list of devices e.g.: [Device <Broadcom ADSL Router>]
    devices=None
    tries=5
    for t in range(tries):
        try:
            devices = upnp.discover(delay=2)
            logger.debug('found IGD devices list {}'.format(devices))
            break
        except Exception as e:
            logger.warning('(Try {} of {}): exception "{}" trying to discover IGD'.format(t, tries, e))
            sleep(.5)

    # Select the IGD
    # alternatively you can select the device directly from the list
    # device = devices[0]
    device = upnp.get_igd()
    logger.debug('selected device {}'.format(device))

    # Get the services available for this device
    # Returns a list of services available for the device
    # e.g.: [<Service (WANPPPConnection) id="WANPPPConnection.1">, ...]
    services= device.get_services()
    logger.debug('found services {}'.format(services))

    # We can now access a specific service on the device by its ID
    # The IDs for the services in this case contain illegal values so we can't access it by an attribute
    # If the service ID didn't contain illegal values we could access it via an attribute like this:
    # service = device.WANPPPConnection

    service=None
    for s in services:
        if s.type_=='WANIPConnection':
            service=s
            logger.debug('found WANPPPConnection service {}'.format(service))
            break

    if service is None:
        raise RuntimeError('Could not find service WANIPConnecton in UPnP router device {}'.format(device))

    # Get the actions available for the service
    # Returns a list of actions for the service:
    #   [<Action name="SetConnectionType">,
    #   <Action name="GetConnectionTypeInfo">,
    #   <Action name="RequestConnection">,
    #   <Action name="ForceTermination">,
    #   <Action name="GetStatusInfo">,
    #   <Action name="GetNATRSIPStatus">,
    #   <Action name="GetGenericPortMappingEntry">,
    #   <Action name="GetSpecificPortMappingEntry">,
    #   <Action name="AddPortMapping">,
    #   <Action name="DeletePortMapping">,
    #   <Action name="GetExternalIPAddress">]
    actions=service.get_actions()
    logger.debug('found actions {}'.format(actions))

    # The action we are looking for is the "AddPortMapping" action
    # Lets see what arguments the action accepts
    # Use the get_input_arguments() or get_output_arguments() method of the action
    # for a list of input / output arguments.
    # Example output of the get_input_arguments method for the "AddPortMapping" action
    # Returns a dictionary:
    # [
    #     {
    #         "name": "NewRemoteHost",
    #         "data_type": "string",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewExternalPort",
    #         "data_type": "ui2",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewProtocol",
    #         "data_type": "string",
    #         "allowed_value_list": [
    #             "TCP",
    #             "UDP"
    #         ]
    #     },
    #     {
    #         "name": "NewInternalPort",
    #         "data_type": "ui2",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewInternalClient",
    #         "data_type": "string",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewEnabled",
    #         "data_type": "boolean",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewPortMappingDescription",
    #         "data_type": "string",
    #         "allowed_value_list": []
    #     },
    #     {
    #         "name": "NewLeaseDuration",
    #         "data_type": "ui4",
    #         "allowed_value_list": []
    #     }
    # ]
    # service.AddPortMapping.get_input_arguments()
    logger.debug('adding port mappings for CLIENT_PORT_RANGE {}'.format(CLIENT_PORT_RANGE))

    my_ip=get_local_ip_address()
    logger.debug('Determined our own IP address is {}'.format(my_ip))

    s = CLIENT_PORT_RANGE.split('-')
    if len(s) != 2:
        raise RuntimeError(
            'port range {} should be of form start-end, e.g. 50100-50200'.format(CLIENT_PORT_RANGE))
    start_port = int(s[0])
    end_port = int(s[1])
    for p in range(start_port, end_port):
        try:
            # Finally, add the new port mapping to the IGD
            # This specific action returns an empty dict: {}
            service.AddPortMapping(
                NewRemoteHost=[],
                NewExternalPort=p,
                NewProtocol='UDP',
                NewInternalPort=p,
                NewInternalClient=my_ip,
                NewEnabled=1,
                NewPortMappingDescription='l2race client mapping',
                NewLeaseDuration=3600
            )
        except Exception as e:
            logger.warning('could not open port {}; caught "{}"'.format(p,e))

