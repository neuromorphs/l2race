# utility methods
import logging
import os

from src.globals import CLIENT_PORT_RANGE

# customized logger with color output
import logging

from src.globals import LOGGING_LEVEL


def my_logger(name):
    logging.basicConfig(level=LOGGING_LEVEL)
    # root = logging.getLogger()
    # root.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
            logging.WARNING))
    logging.addLevelName(
        logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
            logging.ERROR))
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
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
        return
    logging.basicConfig(level=LOGGING_LEVEL)
    # root = logging.getLogger()
    # root.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
            logging.WARNING))
    logging.addLevelName(
        logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
            logging.ERROR))


def bind_socket_to_range(portrange, client_sock):
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
    isbound = False
    for p in range(start_port, end_port):
        try:
            client_sock.bind(('0.0.0.0', p))  # bind to port 0 to get a random free port
            logger.info('bound socket {} to local port {}'.format(client_sock, p))
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

def open_ports():
    import upnpy

    upnp = upnpy.UPnP()

    # Discover UPnP devices on the network
    # Returns a list of devices e.g.: [Device <Broadcom ADSL Router>]
    devices = upnp.discover()

    # Select the IGD
    # alternatively you can select the device directly from the list
    # device = devices[0]
    device = upnp.get_igd()

    # Get the services available for this device
    # Returns a list of services available for the device
    # e.g.: [<Service (WANPPPConnection) id="WANPPPConnection.1">, ...]
    device.get_services()

    # We can now access a specific service on the device by its ID
    # The IDs for the services in this case contain illegal values so we can't access it by an attribute
    # If the service ID didn't contain illegal values we could access it via an attribute like this:
    # service = device.WANPPPConnection

    # We will access it like a dictionary instead:
    service = device['WANPPPConnection.1']

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
    service.get_actions()

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
    service.AddPortMapping.get_input_arguments()

    s = CLIENT_PORT_RANGE.split('-')
    if len(s) != 2:
        raise RuntimeError(
            'client port range {} should be of form start-end, e.g. 50100-50200'.format(portrange))
    start_port = int(s[0])
    end_port = int(s[1])
    isbound = False
    for p in range(start_port, end_port):
        try:
            # Finally, add the new port mapping to the IGD
            # This specific action returns an empty dict: {}
            service.AddPortMapping(
                NewRemoteHost='',
                NewExternalPort=p,
                NewProtocol='UDP',
                NewInternalPort=p,
                NewInternalClient='0.0.0.0',
                NewEnabled=1,
                NewPortMappingDescription='l2race mapping.',
                NewLeaseDuration=3600
            )
        except:
            logger.warning('could not open port {}'.format(p))
    if not isbound:
        raise RuntimeError('could not bind to any port in range {}'.format(CLIENT_PORT_RANGE))

