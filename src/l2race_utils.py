# utility methods
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
