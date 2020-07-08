
import logging
import svglib
from svglib.svglib import svg2rlg
logger = logging.getLogger(__name__)

class track:
    """
    race track
    """

    def __init__(self):
        self.vertices=None
        self.widths=None
        self.drawing=None

        pass

    def draw(self):
        pass

    def load(self,file):
        self.drawing=svg2rlg(file)
        pass
