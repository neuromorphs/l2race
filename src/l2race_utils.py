# utility methods
from src.globals import CLIENT_PORT_RANGE
from src.my_logger import my_logger
logger = my_logger(__name__)

def bind_socket_to_range(portrange, clientSock):
    s = portrange.split('-')
    if len(s) != 2:
        raise RuntimeError(
            'client port range {} should be of form start-end, e.g. 50100-50200'.format(portrange))
    start_port = int(s[0])
    end_port = int(s[1])
    isbound = False
    for p in range(start_port, end_port):
        try:
            clientSock.bind(('0.0.0.0', p))  # bind to port 0 to get a random free port
            logger.info('bound to socket {}'.format(clientSock))
            isbound = True
            break
        except:
            logger.warning('could not bind to port {}'.format(p))
    if not isbound:
        raise RuntimeError('could not bind to any port in range {}'.format(portrange))


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
        raise RuntimeError('could not bind to any port in range {}'.format(portrange))

