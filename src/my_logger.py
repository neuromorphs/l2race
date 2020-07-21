# customized logger with color output
import logging

def my_logger(name):
    logging.basicConfig()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
            logging.WARNING))
    logging.addLevelName(
        logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
            logging.ERROR))
    logger = logging.getLogger(name)
    return logger