import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from src.controllers.neural_mpc_controller_util.globals import *
import json
import time

# from src.controllers.neural_mpc_controller_util.nn_prediction.training.train import train_network


class NeuralNetworkPredictor:

    def __init__(self, model_name = MODEL_NAME):

        print("Initializing nn_predctor with model name {}".format(model_name))

        self.model_name = model_name
     
        #load model
        model_path = 'src/controllers/neural_mpc_controller_util/nn_prediction/models/{}'.format(self.model_name)
        scaler_x_path = 'src/controllers/neural_mpc_controller_util/nn_prediction/models/{}/scaler_x.pkl'.format(self.model_name)
        scaler_y_path = 'src/controllers/neural_mpc_controller_util/nn_prediction/models/{}/scaler_y.pkl'.format(self.model_name)
        nn_settings_path = 'src/controllers/neural_mpc_controller_util/nn_prediction/models/{}/nn_settings.json'.format(self.model_name)

        #chack if model is already trained
        if not os.path.isdir(model_path):
            print("Model {} does not exist. Please train first".format(self.model_name))
            exit()
            # train_network()

        self.model = keras.models.load_model(model_path)
        self.scaler_x = joblib.load(scaler_x_path) 
        self.scaler_y = joblib.load(scaler_y_path) 
        with open(nn_settings_path, 'r') as openfile:
            self.nn_settings = json.load(openfile)
  
        self.predict_delta = self.nn_settings['predict_delta']
        self.normalize_data = self.nn_settings['normalize_data']


    def predict_next_state(self, state, control_input):

        state_and_control = np.append(state, control_input)

        if(self.normalize_data):
            # Normalize input
            state_and_control_normalized = self.scaler_x.transform([state_and_control])
            # Predict
            predictions_normalized = self.model.predict_step(state_and_control_normalized)
            # Denormalize results
            prediction = self.scaler_y.inverse_transform(predictions_normalized)[0]

        else:
            prediction = self.model.predict([state_and_control.tolist()])

        
        if self.predict_delta:
            prediction = state + prediction

        return prediction


    def predict_multiple_states(self, states, control_inputs):
        
        states_and_controls = np.column_stack((states, control_inputs))
   
        if(self.normalize_data):
            # Normalize input
            states_and_controls_normalized = self.scaler_x.transform(states_and_controls)
            # Predict
            predictions_normalized = self.model.predict_step(states_and_controls_normalized)
            # Denormalize results
            predictions = self.scaler_y.inverse_transform(predictions_normalized)

        else:
            predictions = self.model.predict_step(states_and_controls)
     
        if self.predict_delta:
            predictions = states + predictions
        
        return predictions

    #Autoregressively predict multiple trajectories due to given initial states and a list of control sequences
    def predict_multiple_trajectories(self, states, control_inputs):
        
        states_and_controls = np.column_stack((states, control_inputs))

        trajectory_steps = []

        if(self.normalize_data):
            # Normalize input
            states_and_controls_normalized = self.scaler_x.transform(states_and_controls)
            # Predict

            simulated_states = states_and_controls_normalized
            for i in range(15):
                simulated_states = self.model.predict_step(simulated_states)
                trajectory_steps.append(simulated_states)
            # Denormalize results
            predictions = self.scaler_y.inverse_transform(trajectory_steps)

        else:
            predictions = self.model.predict(states_and_controls)

     
        if self.predict_delta:
            predictions = states + predictions
        
        return predictions


if __name__ == '__main__':

    #For direct testing
    predictor = NeuralNetworkPredictor(model_name="Dense-128-128-128-128-uniform-20")


    start = time.time()

    number_of_tests = 1
    test_states = number_of_tests * [ [41.900000000000006,13.600000000000001,0.0,7.0,0.0,0.0,0.0]]
    test_controls = number_of_tests * [  [0.06644491781040185,-0.40862345615368234]]

    trajectories = predictor.predict_multiple_states(test_states,test_controls)
    print(trajectories.shape)

    end = time.time()
    print("TIME FOR {} Trajectoriy".format(number_of_tests))
    print(end - start)
    # print("Next_state", next_state)
