# models that take car_command and car_state and produce next car_state.
# these models are local to the client and are used for control in e.g. MPC
from src.l2race_utils import my_logger
logger = my_logger(__name__)

import car

class linear_extrapolation_model:
    def __init__(self, car:car=None) -> None:
        pass

