import numpy as np


from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

class predictor_ideal:
    def __init__(self, horizon, dt, intermediate_steps=1):
        pass

    def setup(self, initial_state: np.ndarray, prediction_denorm=False):
        pass

    def predict(self, Q: np.ndarray) -> np.ndarray:
        prediction = None
        return prediction

    def update_internal_state(self, Q0):
        pass
