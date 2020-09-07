# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:28:34 2020

@author: Marcin
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from IPython.display import Image

import matplotlib.pyplot as plt

import pandas as pd
import os

import random as rnd
import collections


def get_device():
    """
    Small function to correctly send data to GPU or CPU depending what is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


# Set seeds everywhere required to make results reproducible
def set_seed(args):
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Print parameter count
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def print_parameter_count(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('::: # network all parameters: ' + str(pytorch_total_params))
    print('::: # network trainable parameters: ' + str(pytorch_trainable_params))


def load_pretrained_rnn(net, pt_path, device):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param pt_path: path to .pt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    pre_trained_model = torch.load(pt_path, map_location=device)
    print("Loading Model: ", pt_path)

    pre_trained_model = list(pre_trained_model.items())
    new_state_dict = collections.OrderedDict()
    count = 0
    num_param_key = len(pre_trained_model)
    for key, value in net.state_dict().items():
        if count >= num_param_key:
            break
        layer_name, weights = pre_trained_model[count]
        new_state_dict[key] = weights
        print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1
    net.load_state_dict(new_state_dict)


# Initialize weights and biases - should be only applied if no pretrained net loaded
def initialize_weights_and_biases(net):
    for name, param in net.named_parameters():
        print(name)
        if 'gru' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
        if 'linear' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
                # nn.init.xavier_uniform_(param)
        if 'bias' in name:  # all biases
            nn.init.constant_(param, 0)


def create_rnn_instance(rnn_name=None, inputs_list=None, outputs_list=None, load_rnn=None, path_save=None, device=None):
    if load_rnn is not None and load_rnn != 'last':
        # 1) Find csv with this name if exists load name, inputs and outputs list
        #       if it does not exist raise error
        # 2) Create corresponding net
        # 3) Load parameters from corresponding pt file

        filename = load_rnn
        print('Loading a pretrained RNN with the full name: {}'.format(filename))
        txt_filename = filename + '.txt'
        pt_filename = filename + '.pt'
        txt_path = path_save + txt_filename
        pt_path = path_save + pt_filename

        if not os.path.isfile(txt_path):
            raise ValueError(
                'The corresponding .txt file is missing (information about inputs and outputs) at the location {}'.format(
                    txt_path))
        if not os.path.isfile(pt_path):
            raise ValueError(
                'The corresponding .pt file is missing (information about weights and biases) at the location {}'.format(
                    pt_path))

        f = open(txt_path, 'r')
        lines = f.readlines()
        rnn_name = lines[1].rstrip("\n")
        inputs_list = lines[7].rstrip("\n").split(sep=', ')
        outputs_list = lines[10].rstrip("\n").split(sep=', ')
        f.close()

        print('Inputs to the loaded RNN: {}'.format(', '.join(map(str, inputs_list))))
        print('Outputs from the loaded RNN: {}'.format(', '.join(map(str, outputs_list))))

        # Construct the requested RNN
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)

    elif load_rnn == 'last':
        try:
            import glob
            list_of_files = glob.glob(path_save + '/*.txt')
            txt_path = max(list_of_files, key=os.path.getctime)
        except FileNotFoundError:
            raise ValueError('No information about any pretrained network found at {}'.format(path_save))

        f = open(txt_path, 'r')
        lines = f.readlines()
        rnn_name = lines[1].rstrip("\n")
        pre_rnn_full_name = lines[4].rstrip("\n")
        inputs_list = lines[7].rstrip("\n").split(sep=', ')
        outputs_list = lines[10].rstrip("\n").split(sep=', ')
        f.close()

        print('Inputs to the loaded RNN: {}'.format(', '.join(map(str, inputs_list))))
        print('Outputs from the loaded RNN: {}'.format(', '.join(map(str, outputs_list))))

        # Construct the requested RNN
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)

        pt_path = path_save + pre_rnn_full_name + '.pt'
        if not os.path.isfile(pt_path):
            raise ValueError(
                'The corresponding .pt file is missing (information about weights and biases) at the location {}'.format(
                    pt_path))

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)


    else:  # args.load_rnn is None
        print('No pretrained network specified. I will train a network from scratch.')
        # Construct the requested RNN
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)
        initialize_weights_and_biases(net)

    return net, rnn_name, inputs_list, outputs_list


def create_log_file(rnn_name, inputs_list, outputs_list, path_save):
    rnn_full_name = rnn_name[:4] + str(len(inputs_list)) + 'IN-' + rnn_name[4:] + '-' + str(len(outputs_list)) + 'OUT'

    net_index = 0
    while True:

        txt_path = path_save + rnn_full_name + '-' + str(net_index) + '.txt'
        if os.path.isfile(txt_path):
            pass
        else:
            rnn_full_name += '-' + str(net_index)
            f = open(txt_path, 'w')
            f.write('RNN NAME: \n' + rnn_name + '\n\n')
            f.write('RNN FULL NAME: \n' + rnn_full_name + '\n\n')
            f.write('INPUTS: \n' + ', '.join(map(str, inputs_list)) + '\n\n')
            f.write('OUTPUTS: \n' + ', '.join(map(str, outputs_list)) + '\n\n')
            f.close()
            break

        net_index += 1

    print('Full name given to the currently trained network is {}.'.format(rnn_full_name))
    return rnn_full_name


class Sequence(nn.Module):
    """"
    Our RNN class.
    """
    commands_list = ['dt', 'cmd.auto', 'cmd.steering', 'cmd.throttle', 'cmd.brake',
                     'cmd.reverse']  # Repeat to accept names also without 'cmd.'
    state_variables_list = ['time', 'pos.x', 'pos.y', 'vel.x', 'vel.y', 'speed', 'accel.x', 'accel.y', 'steering_angle',
                            'body_angle', 'yaw_rate', 'drift_angle']

    def __init__(self, rnn_name, inputs_list, outputs_list):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        self.command_inputs = []
        self.states_inputs = []
        for rnn_input in inputs_list:
            if rnn_input in Sequence.commands_list:
                self.command_inputs.append(rnn_input)
            elif rnn_input in Sequence.state_variables_list:
                self.states_inputs.append(rnn_input)
            else:
                s = 'A requested input {} to RNN is neither a command nor a state variable of l2race car model' \
                    .format(rnn_input)
                raise ValueError(s)

        # Check if requested outputs are fine
        for rnn_output in outputs_list:
            if (rnn_output not in Sequence.state_variables_list) and (rnn_output not in Sequence.commands_list):
                s = 'A requested output {} of RNN is neither a command nor a state variable of l2race car model' \
                    .format(rnn_output)
                raise ValueError(s)

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()

        # Get the information about network architecture from the network name
        # Split the names into "LSTM/GRU", "128H1", "64H2" etc.
        names = rnn_name.split('-')
        layers = ['H1', 'H2', 'H3', 'H4', 'H5']
        self.h_size = []  # Hidden layers sizes
        for name in names:
            for index, layer in enumerate(layers):
                if layer in name:
                    # assign the variable with name obtained from list layers.
                    self.h_size.append(int(name[:-2]))

        if not self.h_size:
            raise ValueError('You have to provide the size of at least one hidden layer in rnn name')

        if 'GRU' in names:
            self.rnn_type = 'GRU'
        elif 'LSTM' in names:
            self.rnn_type = 'LSTM'
        else:
            self.rnn_type = 'RNN-Basic'

        # Construct network

        if self.rnn_type == 'GRU':
            self.rnn_cell = [nn.GRUCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.GRUCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        elif self.rnn_type == 'LSTM':
            self.rnn_cell = [nn.LSTMCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.LSTMCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        else:
            self.rnn_cell = [nn.RNNCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.RNNCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))

        self.linear = nn.Linear(self.h_size[-1], len(outputs_list))  # RNN out

        self.layers = nn.ModuleList([])
        for cell in self.rnn_cell:
            self.layers.append(cell)
        self.layers.append(self.linear)

        # Count data samples (=time steps)
        self.sample_counter = 0
        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM
        # Variable keeping the most recent output of RNN
        self.output = None
        # List storing the history of RNN outputs
        self.outputs = []

        # Send the whole RNN to GPU if available, otherwise send it to CPU
        self.to(self.device)

        print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
              .format(self.rnn_type, len(self.h_size), ', '.join(map(str, self.h_size))))
        print('The inputs are (in this order):')
        print('Input state variables: {}'.format(', '.join(map(str, self.states_inputs))))
        print('Input commands: {}'.format(', '.join(map(str, self.command_inputs))))
        print('The outputs are (in this order): {}'.format(', '.join(map(str, outputs_list))))

    def forward(self, predict_len: int, rnn_input_commands, terminate=False, real_time=False):
        """
        Predicts future CartPole states IN "CLOSED LOOP"
        (at every time step prediction for the next time step is done based on CartPole state
        resulting from the previous prediction; only control input is provided from the ground truth at every step)
        """

        # For number of time steps given in predict_len predict the state of the CartPole
        # At every time step RNN get as its input the ground truth value of control input
        # BUT instead of the ground truth value of CartPole state
        # it gets the result of the prediction for the last time step
        for iteration in range(predict_len):

            # Concatenate the previous prediction and current control input to the input to RNN for a new time step
            if real_time:
                input_t = torch.cat((self.output, rnn_input_commands.squeeze(0)), 1)
            else:
                input_t = torch.cat((self.output, rnn_input_commands[self.sample_counter, :]), 1)

            # Propagate input through RNN layers

            if self.rnn_type == 'LSTM':
                self.h[0], self.c[0] = self.layers[0](input_t, (self.h[0], self.c[0]))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
            else:
                self.h[0] = self.layers[0](input_t, self.h[0])
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])

            self.output = self.layers[-1](self.h[-1])

            # Append the output to the outputs history list
            self.outputs += [self.output]
            # Count number of samples
            self.sample_counter = self.sample_counter + 1

        # if terminate=True transform outputs history list to a Pytorch tensor and return it
        # Otherwise store the outputs internally as a list in the RNN instance
        if terminate:
            self.outputs = torch.stack(self.outputs, 1)
            return self.outputs
        else:
            return self.output

    def reset(self):
        """
        Reset the network (not the weights!)
        """
        self.sample_counter = 0
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)
        self.output = None
        self.outputs = []

    def initialize_sequence(self, rnn_input, warm_up_len=256, stack_output=True, all_input=False):

        """
        Predicts future CartPole states IN "OPEN LOOP"
        (at every time step prediction for the next time step is done based on the true CartPole state)
        """

        # If in training mode we will only run this function during the first several (warm_up_len) data samples
        # Otherwise we run it for the whole input
        if not all_input:
            starting_input = rnn_input[:warm_up_len, :, :]
        else:
            starting_input = rnn_input

        # Initialize hidden layers - this change at every call as the batch size may vary
        for i in range(len(self.h_size)):
            self.h[i] = torch.zeros(starting_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)
            self.c[i] = torch.zeros(starting_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)

        # The for loop takes the consecutive time steps from input plugs them into RNN and save the outputs into a list
        # THE NETWORK GETS ALWAYS THE GROUND TRUTH, THE REAL STATE OF THE CARTPOLE, AS ITS INPUT
        # IT PREDICTS THE STATE OF THE CARTPOLE ONE TIME STEP AHEAD BASED ON TRUE STATE NOW
        for iteration, input_t in enumerate(starting_input.chunk(starting_input.size(0), dim=0)):

            # Propagate input through RNN layers
            if self.rnn_type == 'LSTM':
                self.h[0], self.c[0] = self.layers[0](input_t.squeeze(0), (self.h[0], self.c[0]))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
            else:
                self.h[0] = self.layers[0](input_t.squeeze(0), self.h[0])
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])
            self.output = self.layers[-1](self.h[-1])

            self.outputs += [self.output]
            self.sample_counter = self.sample_counter + 1

        # In the train mode we want to continue appending the outputs by calling forward function
        # The outputs will be saved internally in the network instance as a list
        # Otherwise we want to transform outputs list to a tensor and return it
        if stack_output:
            outputs_return = torch.stack(self.outputs, 1)
            return outputs_return
        else:
            return self.output


def norm(x):
    m = np.mean(x)
    s = np.std(x)
    y = (x - m) / s
    return y


class Dataset(data.Dataset):
    def __init__(self, df, labels, args):
        'Initialization'
        self.data = df
        self.labels = labels
        self.args = args

        # Hyperparameters
        self.seq_len = args.seq_len  # Sequence length

    def __len__(self):
        'Total number of samples'

        # speed optimized, you only have to get sequencies
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len, :], self.labels[idx:idx + self.seq_len]


def normalize(dat, mean, std):
    rep = int(dat.shape[-1] / mean.shape[
        0])  # there is 1 mean for each input sensor value, repeat it for each element in sequence in data
    mean = np.tile(mean, rep)
    std = np.tile(std, rep)
    dat = (dat - mean) / std
    return dat


def unnormalize(dat, mean, std):
    rep = int(dat.shape[-1] / mean.shape[0])  # results in cw_len to properly tile for applying to dat.
    # there is 1 mean for each input sensor value, repeat it for each element in sequence in data
    mean = np.tile(mean, rep)
    std = np.tile(std, rep)
    dat = dat * std + mean
    return dat


def save_normalization(save_path, tr_mean, tr_std, lab_mean, lab_std):
    fn_base = os.path.splitext(save_path)[0]
    print("\nSaving normalization parameters to " + str(fn_base) + '-XX.pt')
    norm = {
        'tr_mean': tr_mean,
        'tr_std': tr_std,
        'lab_mean': lab_mean,
        'lab_std': lab_std,
    }
    torch.save(norm, str(fn_base + '-norm.pt'))


def load_normalization(save_path):
    fn_base = os.path.splitext(save_path)[0]
    print("\nLoading normalization parameters from ", str(fn_base))
    norm = torch.load(fn_base + '-norm.pt')
    return norm['tr_mean'], norm['tr_std'], norm['lab_mean'], norm['lab_std']


def computeNormalization(dat: np.array):
    '''
    Computes the special normalization of our data
    Args:
        dat: numpy array of data arranged as [sample, sensor/control vaue]

    Returns:
        mean and std, each one is a vector of values
    '''
    # Collect desired prediction
    # All terms are weighted equally by the loss function (either L1 or L2),
    # so we need to make sure that we don't have values here that are too different in range
    # Since the derivatives are computed on normalized data, but divided by the delta time in ms,
    # we need to normalize the derivatives too. (We include the delta time to make more physical units of time
    # so that the values don't change with different sample rate.)
    m = np.mean(dat, 0)
    s = np.std(dat, 0)
    return m, s


# def load_data(filepath, args, savepath=None, save_normalization_parameters=False):
import sys

sys.path.append('../../')
from src.track import find_hit_position, meters2pixels, pixels2meters, track


def load_data(filepath, inputs_list, outputs_list, args):
    '''
    Loads dataset from CSV file
    Args:
        filepath: path to CSV file
        seq_len: number of samples in time input to RNN
        stride: step over data in samples between samples
        medfilt: median filter window, 0 to disable
        test_plot: test_plot.py file requires other way of loading data

    Returns:
        Unnormalized numpy arrays
        input_data:  indexed by [sample, sequence, # sensor inputs * cw_len]
        input dict:  dictionary of input data names with key index into sensor index value
        targets:  indexed by [sample, sequence, # output sensor values * pw_len]
        target dict:  dictionary of target data names with key index into sensor index value
        actual: raw input data indexed by [sample,sensor]
        actual dict:  dictionary of actual inputs with key index into sensor index value
        mean_features, std_features, mean_targets_data, std_targets_data: the means and stds of training and raw_targets data.
            These are vectors with length # of sensorss

        The last dimension of input_data and targets is organized by
         all sensors for sample0, all sensors for sample1, .....all sensors for sampleN, where N is the prediction length
    '''

    # Hyperparameters

    # Load dataframe
    print('loading data from ' + str(filepath))
    df = pd.read_csv(filepath, comment='#')

    print('processing data to generate sequences')

    # Calculate time difference between time steps and at it to data frame
    time = df['time'].to_numpy()
    deltaTime = np.diff(time)
    deltaTime = np.insert(deltaTime, 0, 0)
    df['dt'] = deltaTime

    if args.extend_df:
        # Calculate distance to the track edge in front of the car and add it to the data frame
        # Get used car name
        import re
        s = str(pd.read_csv(filepath, skiprows=5, nrows=1))
        track_name = re.search('"(.*)"', s).group(1)
        media_folder_path = '../../media/tracks/'
        my_track = track(track_name=track_name, media_folder_path=media_folder_path)

        def calculate_hit_distance(row):
            return my_track.get_hit_distance(angle=row['body_angle'], x_car=row['pos.x'], y_car=row['pos.y'])

        df['hit_distance'] = df.apply(calculate_hit_distance, axis=1)

        def nearest_waypoint_idx(row):
            return my_track.get_nearest_waypoint_idx(x=row['pos.x'], y=row['pos.y'])

        df['nearest_waypoint_idx'] = df.apply(nearest_waypoint_idx, axis=1)

        max_idx = max(df['nearest_waypoint_idx'])

        def get_nth_next_waypoint_x(row, n: int):
            return pixels2meters(my_track.waypoints_x[int(row['nearest_waypoint_idx'] + n) % max_idx])

        def get_nth_next_waypoint_y(row, n: int):
            return pixels2meters(my_track.waypoints_y[int(row['nearest_waypoint_idx'] + n) % max_idx])

        df['first_next_waypoint.x'] = df.apply(get_nth_next_waypoint_x, axis=1, args=(1,))
        df['first_next_waypoint.y'] = df.apply(get_nth_next_waypoint_y, axis=1, args=(1,))
        df['fifth_next_waypoint.x'] = df.apply(get_nth_next_waypoint_x, axis=1, args=(5,))
        df['fifth_next_waypoint.y'] = df.apply(get_nth_next_waypoint_y, axis=1, args=(5,))
        df['twentieth_next_waypoint.x'] = df.apply(get_nth_next_waypoint_x, axis=1, args=(20,))
        df['twentieth_next_waypoint.y'] = df.apply(get_nth_next_waypoint_y, axis=1, args=(20,))

    # Get Raw Data
    # x = df[args.features_list]
    # u = df[args.commands_list]
    # y = df[args.targets_list]
    inputs = df[inputs_list]
    outputs = df[outputs_list]
    features = np.array(inputs)[:-1]
    targets = np.array(outputs)[1:]

    # # @Nikhil It seems there is slightly different slicing convention for pandas DataFrame - here it was crashing
    # # I don't know how to do it correctly so I convert it to numpy. You are welcome to change it.
    # states = np.array(x)[:-1]
    # control = np.array(u)[:-1]
    # targets = np.array(y)[1:]
    #
    #
    # features = np.hstack((np.array(states), np.array(control)))
    # # features = torch.from_numpy(features).float()
    #
    # targets = np.array(targets)
    # # targets = torch.from_numpy(targets).float()
    # TODO : Compare the dimensions of features, and targets(By Nikhil) with that of raw_features, raw_targets(By Marcin)
    #       and transpose accordingly if required
    # Good job Nikhil! I like your approach!

    # # The normalization of data - I am not sure how necessary it is
    # # If you uncomment this lines please make sure that you unnormalize data before plugging them into ghost car
    # # compute normalization of data now
    # mean_features, std_features = computeNormalization(features)
    # mean_targets, std_targets = computeNormalization(targets)
    #
    # if save_normalization_parameters:  # We only save normalization for train set - all the other sets we normalize withr respect to this set
    #     save_normalization(savepath, mean_features, std_features, mean_targets, std_targets)
    #     mean_train_features, std_train_features, mean_train_targets, std_train_targets = \
    #         mean_features, std_features, mean_targets, std_targets
    # else:
    #     mean_train_features, std_train_features, mean_train_targets, std_train_targets \
    #         = load_normalization(savepath)
    #
    # features = normalize(features, mean_train_features, std_train_features)
    # targets = normalize(targets, mean_train_targets, std_train_targets)

    # Version with normlaization
    # return features, targets, mean_features, std_features, mean_targets, std_targets
    return features, targets

# def plot_results(net, args, val_savepathfile):
#     """
#     This function accepts RNN instance, arguments and CartPole instance.
#     It runs one random experiment with CartPole,
#     inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
#     """
#     # Reset the internal state of RNN cells, clear the output memory, etc.
#     net.reset()
#
#     # Generates ab CartPole  experiment and save its data
#     dev_features, dev_targets, _, _, _, _ = \
#         load_data(val_file, args, savepath)
#
#     dev_set = Dataset(dev_features, dev_targets, args)
#
#     # Format the experiment data
#     # test_set[0] means that we take one random experiment, first on the list
#     # The data will be however anyway generated on the fly and is in general not reproducible
#     # TODO: Make data reproducable: set seed or find another solution
#     features, targets = test_set[0]
#
#     # Add empty dimension to fit the requirements of RNN input shape
#     # (in fact we add dimension for batches - for testing we use only one batch)
#     features = features.unsqueeze(0)
#
#     # Convert Pytorch tensors to numpy matrices to inspect them - just for debugging purpose.
#     # Variable explorers of IDEs are often not compatible with Pytorch format
#     # features_np = features.detach().numpy()
#     # targets_np = targets.detach().numpy()
#
#     # Further modifying the input and output form to fit RNN requirements
#     # If GPU available we send features to GPU
#     if torch.cuda.is_available():
#         features = features.float().cuda().transpose(0, 1)
#         targets = targets.float()
#     else:
#         features = features.float().transpose(0, 1)
#         targets = targets.float()
#
#     # From features we extract control input and save it as a separate vector on the cpu
#     u_effs = features[:, :, -1].cpu()
#     # We shift it by one time step and double the last entry to keep the size unchanged
#     u_effs = u_effs[1:]
#     u_effs = np.append(u_effs, u_effs[-1])
#
#     # Set the RNN in evaluation mode
#     net = net.eval()
#     # During several first time steps we let hidden layers adapt to the input data
#     # train=False says that all the input should be used for initialization
#     # -> we predict always only one time step ahead of time based on ground truth data
#     predictions = net.initialize_sequence(features, train=False)
#
#     # reformat the output of RNN to a form suitable for plotting the results
#     # y_pred are prediction from RNN
#     y_pred = predictions.squeeze().cpu().detach().numpy()
#     # y_target are expected prediction from RNN, ground truth
#     y_target = targets.squeeze().cpu().detach().numpy()
#
#     # Get the time axes
#     t = np.arange(0, y_target.shape[0]) * args.dt
#
#     # Get position over time
#     xp = y_pred[args.warm_up_len:, 0]
#     xt = y_target[:, 0]
#
#     # Get velocity over time
#     vp = y_pred[args.warm_up_len:, 1]
#     vt = y_target[:, 1]
#
#     # get angle theta of the Pole
#     tp = y_pred[args.warm_up_len:, 2] * 180.0 / np.pi  # t like theta
#     tt = y_target[:, 2] * 180.0 / np.pi
#
#     # Get angular velocity omega of the Pole
#     op = y_pred[args.warm_up_len:, 3] * 180.0 / np.pi  # o like omega
#     ot = y_target[:, 3] * 180.0 / np.pi
#
#     # Create a figure instance
#     fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots
#
#     # %matplotlib inline
#     # Plot position
#     axs[0].set_ylabel("Position (m)", fontsize=18)
#     axs[0].plot(t, xt, 'k:', markersize=12, label='Ground Truth')
#     axs[0].plot(t[args.warm_up_len:], xp, 'b', markersize=12, label='Predicted position')
#     axs[0].tick_params(axis='both', which='major', labelsize=16)
#
#     # Plot velocity
#     axs[1].set_ylabel("Velocity (m/s)", fontsize=18)
#     axs[1].plot(t, vt, 'k:', markersize=12, label='Ground Truth')
#     axs[1].plot(t[args.warm_up_len:], vp, 'g', markersize=12, label='Predicted velocity')
#     axs[1].tick_params(axis='both', which='major', labelsize=16)
#
#     # Plot angle
#     axs[2].set_ylabel("Angle (deg)", fontsize=18)
#     axs[2].plot(t, tt, 'k:', markersize=12, label='Ground Truth')
#     axs[2].plot(t[args.warm_up_len:], tp, 'c', markersize=12, label='Predicted angle')
#     axs[2].tick_params(axis='both', which='major', labelsize=16)
#
#     # Plot angular velocity
#     axs[3].set_ylabel("Angular velocity (deg/s)", fontsize=18)
#     axs[3].plot(t, ot, 'k:', markersize=12, label='Ground Truth')
#     axs[3].plot(t[args.warm_up_len:], op, 'm', markersize=12, label='Predicted velocity')
#     axs[3].tick_params(axis='both', which='major', labelsize=16)
#
#     # Plot motor input command
#     axs[4].set_ylabel("motor (N)", fontsize=18)
#     axs[4].plot(t, u_effs, 'r', markersize=12, label='motor')
#     axs[4].tick_params(axis='both', which='major', labelsize=16)
#
#     # # Plot target position
#     # axs[5].set_ylabel("position target", fontsize=18)
#     # axs[5].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['PositionTarget'], 'k')
#     # axs[5].tick_params(axis='both', which='major', labelsize=16)
#
#     axs[4].set_xlabel('Time (s)', fontsize=18)
#
#     plt.show()
#     # Save figure to png
#     fig.savefig('my_figure.png')
#     Image('my_figure.png')
