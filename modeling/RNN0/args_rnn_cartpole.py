# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse

path_save = './save/'
RNN_name = 'GRU-64H1-64H2'

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # Defining the model
    parser.add_argument('--rnn_name', default=RNN_name, type=str,
                        help='Name defining the RNN.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM])-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')
    parser.add_argument('--inputs_list', nargs="+", default=['throttle', 'brake', 'cmd.steering', 'pos.x', 'pos.y', 'vel.x', 'vel.y', 'body_angle'],
                        help='List of inputs to RNN')
    parser.add_argument('--outputs_list', nargs="+", default=['pos.x', 'pos.y', 'body_angle'],
                        help='List of outputs from RNN')
    parser.add_argument('close_loop_for', nargs='?', default=['pos.x', 'pos.y', 'body_angle'],
                        help='In RNN forward function this features will be fed beck from output to input')

    parser.add_argument('--warm_up_len',    default=512,         type=int,    help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--seq_len', default=512+512+1, type=int, help='Number of timesteps in a sequence')

    # Training parameters
    parser.add_argument('--epoch_len',      default=2e3,        type=int,    help='How many sequences are fed in NN during one epoch of training')
    parser.add_argument('--num_epochs',     default=1,         type=int,    help='Number of epochs of training')
    parser.add_argument('--batch_size',     default=128,         type=int,    help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-4, type=float, help='Learning rate')
    parser.add_argument('--path_save', default=path_save, type=str,
                        help='Path where to save/ from where to load models')
    
    parser.add_argument('--num_workers',    default=1,          type=int,    help='Number of workers to produce data from data loaders')



    my_args = parser.parse_args()

    # Adjust args in place to give user more freedom in his input and check it
    commands_list = ['dt', 'cmd.auto', 'cmd.steering', 'cmd.throttle', 'cmd.brake', 'cmd.reverse'] # Repeat to accept names also without 'cmd.'
    state_variables_list = ['time', 'hit_distance', 'pos.x', 'pos.y', 'vel.x', 'vel.y', 'speed', 'accel.x', 'accel.y', 'steering_angle', 'body_angle', 'yaw_rate', 'drift_angle']

    # If user provided command names without cmd. add it.
    for index, rnn_input in enumerate(my_args.inputs_list):
        if rnn_input == 'throttle':
            my_args.inputs_list[index] = 'cmd.throttle'
        if rnn_input == 'auto':
            my_args.inputs_list[index] = 'cmd.auto'
        if rnn_input == 'steering':
            my_args.inputs_list[index] = 'cmd.steering'
        if rnn_input == 'brake':
            my_args.inputs_list[index] = 'cmd.brake'
        if rnn_input == 'reverse':
            my_args.inputs_list[index] = 'cmd.reverse'

    # Make sure that inputs are ordered:
    # First state variables then commands, otherwise alphabetically
    states_inputs = []
    command_inputs = []
    for rnn_input in my_args.inputs_list:
        if rnn_input in commands_list:
            command_inputs.append(rnn_input)
        elif rnn_input in state_variables_list:
            states_inputs.append(rnn_input)
        else:
            s = 'A requested input {} to RNN is neither a command nor a state variable of l2race car model' \
                .format(rnn_input)
            raise ValueError(s)
    my_args.inputs_list = sorted(command_inputs)+sorted(states_inputs)

    # Check if arguments for feeding in closed loop are correct
    for rnn_input in my_args.close_loop_for:
        if (rnn_input not in my_args.inputs_list) or (rnn_input not in my_args.outputs_list):
            raise ValueError('The variable {} you requested to be fed back from RNN output to its input '
                             'is missing on the inputs or outputs list'.format(rnn_input))






    return my_args
