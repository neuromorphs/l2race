# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""

import argparse

path_save = './save/'
TRAIN_file_name = [#'../../data/oval_easy_14_rounds.csv',
                   '../../data/track_1_10_rounds.csv',
                   '../../data/track_2_10_rounds.csv',
                   '../../data/track_3_2_rounds.csv',
                   '../../data/track_3_3_rounds.csv',
                   '../../data/track_3_3_rounds_1.csv',
                   '../../data/track_3_8_rounds.csv.csv' , '../../data/empty_4min.csv', '../../data/empty_6min.csv']
VAL_file_name = '../../data/oval_easy_12_rounds.csv'
RNN_name = 'GRU-256H1-256H2'
# inputs_list = ['dt', 'command.reverse', 'command.brake', 'command.steering', 'command.throttle', 'body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
# inputs_list = ['command.brake', 'command.steering', 'command.throttle', 'body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y',
#                'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
inputs_list = ['command.brake', 'command.steering', 'command.throttle', 'body_angle_deg', 'position_m.x', 'position_m.y',
               'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
# outputs_list = ['body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
# outputs_list = ['body_angle_deg', 'position_m.x', 'position_m.y']
# outputs_list = ['body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
outputs_list = ['body_angle_deg', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
# closed_loop_list = ['body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
closed_loop_list = ['body_angle_deg', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']

# closed_loop_list = ['body_angle_deg', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y']
# closed_loop_list = ['body_angle.cos', 'body_angle.sin', 'position_m.x', 'position_m.y']

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # Defining the model
    parser.add_argument('--rnn_name', nargs='?', const=RNN_name, default=None, type=str,
                        help='Name defining the RNN.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM])-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')
    parser.add_argument('--train_file_name', default=TRAIN_file_name, type=str,
                        help='File name of the recording to be used for training the RNN'
                             'e.g. oval_easy.csv ')
    parser.add_argument('--val_file_name', default=VAL_file_name, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--inputs_list', nargs="?", default=None, const=inputs_list,
                        help='List of inputs to RNN')
    parser.add_argument('--outputs_list', nargs="?", default=None, const=outputs_list,
                        help='List of outputs from RNN')
    parser.add_argument('--close_loop_for', nargs='?', default=None, const=closed_loop_list,
                        help='In RNN forward function this features will be fed beck from output to input')
    parser.add_argument('--load_rnn', nargs='?', default=None, const='last', type=str,
                        help='Full name defining the RNN which should be loaded without .csv nor .pt extension'
                             'e.g. GRU-8IN-64H1-64H2-3OUT-1')
    parser.add_argument("--extend_df", action='store_true',
                        help="Extend loaded data with distance to 'hit-point' and positions of 1st, 5th and 20th nearest waypoints")
    parser.add_argument("--do_not_normalize", action='store_true',
                        help="Normalize data for trainig and inference.")
    parser.add_argument("--cheat_dt", action='store_true',
                        help="Give RNN during training a true (future) dt.")

    parser.add_argument('--warm_up_len', default=1, type=int, help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--seq_len', default=5, type=int, help='Number of timesteps in a sequence')

    # Training parameters
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
    parser.add_argument('--path_save', default=path_save, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--normalize', default=True, type=bool, help='Make all data between 0 and 1')

    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to produce data from data loaders')

    my_args = parser.parse_args()

    # Adjust args in place to give user more freedom in his input and check it
    commands_list = ['dt', 'command.autodrive_enabled', 'command.steering', 'command.throttle', 'command.brake',
                     'command.reverse']  # Repeat to accept names also without 'cmd.'
    state_variables_list = ['time', 'hit_distance', 'position_m.x', 'position_m.y', 'velocity_m_per_sec.x', 'velocity_m_per_sec.y', 'speed_m_per_sec', 'accel_m_per_sec_2.x', 'accel_m_per_sec_2.y',
                            'steering_angle_deg', 'body_angle_deg', 'body_angle.cos', 'body_angle.sin', 'yaw_rate_deg_per_sec',
                            'drift_angle_deg']

    if my_args.inputs_list is not None:
        # If user provided command names without cmd. add it.
        for index, rnn_input in enumerate(my_args.inputs_list):
            if rnn_input == 'throttle':
                my_args.inputs_list[index] = 'command.throttle'
            if rnn_input == 'auto':
                my_args.inputs_list[index] = 'command.autodrive_enabled'
            if rnn_input == 'steering':
                my_args.inputs_list[index] = 'command.steering'
            if rnn_input == 'brake':
                my_args.inputs_list[index] = 'command.brake'
            if rnn_input == 'reverse':
                my_args.inputs_list[index] = 'command.reverse'

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
        my_args.inputs_list = sorted(command_inputs) + sorted(states_inputs)

    # Make sure that inputs are ordered:
    # First state variables then commands, otherwise alphabetically
    if my_args.outputs_list is not None:
        states_outputs = []
        command_outputs = []
        for rnn_output in my_args.outputs_list:
            if rnn_output in commands_list:
                command_outputs.append(rnn_output)
            elif rnn_output in state_variables_list:
                states_outputs.append(rnn_output)
            else:
                s = 'A requested output {} to RNN is neither a command nor a state variable of l2race car model' \
                    .format(rnn_output)
                raise ValueError(s)
        my_args.outputs_list = sorted(command_outputs) + sorted(states_outputs)

    # Check if arguments for feeding in closed loop are correct
    if (my_args.close_loop_for is not None) and (my_args.inputs_list is not None) and (
            my_args.outputs_list is not None):
        for rnn_input in my_args.close_loop_for:
            if (rnn_input not in my_args.inputs_list) or (rnn_input not in my_args.outputs_list):
                raise ValueError('The variable {} you requested to be fed back from RNN output to its input '
                                 'is missing on the inputs or outputs list'.format(rnn_input))

    return my_args
