import numpy as np

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
