"""
Utilities for RNNs in l2race challenge
Adapted from : https://github.com/SensorsINI/CartPoleSimulation

Ref. to dev branch for improved version. 


Date created : 30th August
TODO : 
1. Fix the dimensions of Sequence class as per l2race car state recording
2. Create a data object and try forward, and initialize sequence method on that object
3. Cutting sequences for feeding into RNN

"""

from src.car_state import car_state
from src.globals import *
from src.car import car
import glob


import glob
import numpy as np
import pandas as pd
from random import randrange
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Data:
    """
    A class for simplified loading of simulator data for use with pysindy
    Supports loading data from single or multiple .csv files
    ...
    Attributes
    ----------
    x : np.array or list of np.arrays
        values of features listed in FEATURES variable 
    x_dot : np.array or list of np.arrays
        precalculated derivatives of features listed in DERIVATIVES variable
    u : np.array or list of np.arrays
        values of car commands specified in COMMANDS variable
    t : np.array or list of np.arrays
        timestamps
    multiple_recordings: bool
        specifies whether data is loaded from single or multiple files
        if True, data is stored as lists of np.arrays
    """

    def __init__(self, data_dir, features_list, commands_list):
        """
        Parameters
        ----------
        data_dir: str
            directory containing .csv files
        """

        file_list = glob.glob(data_dir + '/*.csv')

        if len(file_list) == 0:
            raise Exception("Data directory is empty!")

        elif len(file_list) == 1:
            multiple_recordings = False

            full_path = data_dir + '/' + file_list[0]
            data = pd.read_csv(full_path, comment='#')
            data = data.drop_duplicates('time')
            
            t = data['time'].values
            u = data[commands_list].values
            x = data[features_list].values

        else:
            multiple_trajectories = True

            t = []
            x = []
            u = []
            for filename in file_list:
                full_path = data_dir + '/' + filename
                data = pd.read_csv(full_path, comment='#')
                data = data.drop_duplicates('time') # removes "duplicate" timestamps 

                t_single = data['time']
                u_single = data[commands_list]
                x_single = data[features_list]

                t.append(t_single.values)
                x.append(x_single.values)
                u.append(u_single.values) 

        self.x = x
        self.u = u
        self.t = t
        self.multiple_recordings = multiple_recordings




class Dataset(data.Dataset):
    """
    This is a Dataset class providing a proper data format for Pytorch applications
    It inherits from the standard Pytorch dataset class
    """

    def __init__(self,data_dir, args, train=True):
       
        """Initialization"""
             """
        Parameters
        ----------
        data_dir: str
            directory containing .csv files
        """
        # # Take data set of different size for training and testing of RNN
        # if train:
        #     self.exp_len = args.exp_len_train
        # else:
        #     self.exp_len = args.exp_len_test
        '''
        TODO: Add features_list and commands_list to args
        '''
        data = Data(data_dir, args.features_list, args.commands_list)

        # # Recalculate simulation time step from milliseconds to seconds
        # self.dt = args.dt / 1000.0  # s
        # self.epoch_len = args.epoch_len

    def __len__(self):
        """
        Total number of samples.
        In this implementation it is meaningless however needs to be preserved due to Pytorch requirements
        """

        return int(self.epoch_len)

    def __getitem__(self, idx):
        
        """
        When called this function reads the recording file in a form suitable to be used to train/test RNN
        (inputs list to RNN in features array and expected outputs from RNN )
        """

        features_list = ['pos.x', 'pos.y','vel.x','vel.y','steering_angle','body_angle','yaw_rate','drift_angle']
        commands_list = ['cmd.throttle', 'cmd.steering', 'cmd.brake', 'cmd.reverse']

        data = Data(data_dir, features_list, commands_list)
        # "features" is the array of inputs to the RNN, it consists of states of the car and control input
        # "targets" is the array of car states one time step ahead of "features" at the same index.
        # "targets[i]" is what we expect our network to predict given features[i]
        # file_length: length of the selected recording file
        file_length = data.x[0].shape[0]

        states = data.x[0][:(file_length-1),:]
        control = data.u[0][1:,:]
        target = data.x[0][1:,:]
 
        features = np.hstack((np.array(states), np.array(control)))
        features = torch.from_numpy(features).float()

        targets = np.array(targets)
        targets = torch.from_numpy(targets).float()

        return features, targets


class Sequence(nn.Module):
    """"
    Our RNN class.(Adapted from cartpole project)
    """

    def __init__(self, args):
        super(Sequence, self).__init__()
      
        """Initialization of an RNN instance"""
        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()
        # Save args (default of from terminal line)
        self.args = args
        # Initialize RNN layers
        # TODO : Remove hardcoding and take the args and calculate no. of inputs
        self.gru1 = nn.GRUCell(12, args.h1_size)  # RNN accepts 12 inputs: car state (8) and control input(4) at time t
        self.gru2 = nn.GRUCell(args.h1_size, args.h2_size)
        self.linear = nn.Linear(args.h2_size, 8)  # RNN out car state for t+1 ?
        # Count data samples (=time steps)
        self.sample_counter = 0
        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h_t = None
        self.h_t2 = None
        # Variable keeping the most recent output of RNN
        self.output = None
        # List storing the history of RNN outputs
        self.outputs = []

        # Send the whole RNN to GPU if available, otherwise send it to CPU
        self.to(self.device)

    def forward(self, predict_len: int, input, terminate=False):
        """
        Predicts future car states IN "CLOSED LOOP"
        (at every time step prediction for the next time step is done based on car state
        resulting from the previous prediction; only control input is provided from the ground truth at every step)
        """
        # From input to RNN (car state + control input) get control input
        u_effs = input[:, :, -1]
        # For number of time steps given in predict_len predict the state of the car
        # At every time step RNN get as its input the ground truth value of control input
        # BUT instead of the ground truth value of car state
        # it gets the result of the prediction for the last time step
        for i in range(predict_len):
            # Concatenate the previous prediction and current control input to the input to RNN for a new time step
            input_t = torch.cat((self.output, u_effs[self.sample_counter, :].unsqueeze(1)), 1)
            # Propagate input through RNN layers
            self.h_t = self.gru1(input_t, self.h_t)
            self.h_t2 = self.gru2(self.h_t, self.h_t2)
            self.output = self.linear(self.h_t2)
            # Append the output to the outputs history list
            self.outputs += [self.output]
            # Count number of samples
            self.sample_counter = self.sample_counter + 1

        # if terminate=True transform outputs history list to a Pytorch tensor and return it
        # Otherwise store the outputs internally as a list in the RNN instance
        if terminate:
            self.outputs = torch.stack(self.outputs, 1)
            return self.outputs

    def reset(self):
        """
        Reset the network (not the weights!)
        """
        self.sample_counter = 0
        self.h_t = None
        self.h_t2 = None
        self.output = None
        self.outputs = []

    def initialize_sequence(self, input, train=True):

        """
        Predicts future car states IN "OPEN LOOP"
        (at every time step prediction for the next time step is done based on the true car state)
        """

        # If in training mode we will only run this function during the first several (args.warm_up_len) data samples
        # Otherwise we run it for the whole input
        if train:
            starting_input = input[:self.args.warm_up_len, :, :]
        else:
            starting_input = input

        # Initialize hidden layers
        self.h_t = torch.zeros(starting_input.size(1), self.args.h1_size, dtype=torch.float).to(self.device)
        self.h_t2 = torch.zeros(starting_input.size(1), self.args.h2_size, dtype=torch.float).to(self.device)

        # The for loop takes the consecutive time steps from input plugs them into RNN and save the outputs into a list
        # THE NETWORK GETS ALWAYS THE GROUND TRUTH, THE REAL STATE OF THE CAR, AS ITS INPUT
        # IT PREDICTS THE STATE OF THE CAR ONE TIME STEP AHEAD BASED ON TRUE STATE NOW
        for i, input_t in enumerate(starting_input.chunk(starting_input.size(0), dim=0)):
            self.h_t = self.gru1(input_t.squeeze(0), self.h_t)
            self.h_t2 = self.gru2(self.h_t, self.h_t2)
            self.output = self.linear(self.h_t2)
            self.outputs += [self.output]
            self.sample_counter = self.sample_counter + 1

        # In the train mode we want to continue appending the outputs by calling forward function
        # The outputs will be saved internally in the network instance as a list
        # Otherwise we want to transform outputs list to a tensor and return it
        if not train:
            self.outputs = torch.stack(self.outputs, 1)
            return self.outputs
