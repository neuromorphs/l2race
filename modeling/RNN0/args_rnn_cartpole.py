# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse

path_save_model = './save/' + 'MyNet' + '.pt'
path_pretrained = './save/' + 'MyNetPre' + '.pt'
RNN_name = 'LSTM-64H1-64H2'

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # Defining the model
    parser.add_argument('--RNN_name', default=RNN_name, type=str,
                        help='Name defining the RNN.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM])-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2')
    parser.add_argument('--inputs_list', nargs="+", default=['dt', 'throttle', 'brake', 'speed'],
                        help='List of inputs to RNN')
    parser.add_argument('--outputs_list', nargs="+", default=['speed'],
                        help='List of outputs from RNN')

    parser.add_argument('--warm_up_len',    default=512,         type=int,    help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--seq_len', default=512+512+1, type=int, help='Number of timesteps in a sequence')

    # Training parameters
    parser.add_argument('--epoch_len',      default=2e4,        type=int,    help='How many sequences are fed in NN during one epoch of training')
    parser.add_argument('--num_epochs',     default=10,         type=int,    help='Number of epochs of training')
    parser.add_argument('--batch_size',     default=128,         type=int,    help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-4, type=float, help='Learning rate')
    parser.add_argument('--path_save_model', default=path_save_model, type=str,
                        help='Path where to save currently trained model')
    
    parser.add_argument('--num_workers',    default=1,          type=int,    help='Number of workers to produce data from data loaders')

    parser.add_argument('--path_pretrained',           default=path_pretrained, type=str,    help='Path from where to load a pretrained model')

    # The below lines should be redundant and deleted later
    # parser.add_argument('--features_list',   nargs="+",  default=['pos.x', 'pos.y','vel.x','vel.y','steering_angle','body_angle','yaw_rate','drift_angle'],    help='List of features')
    # parser.add_argument('--commands_list', nargs="+", default=['cmd.throttle', 'cmd.steering', 'cmd.brake', 'cmd.reverse'],       help='List of commands')
    # parser.add_argument('--targets_list', nargs="+", default=['pos.x', 'pos.y','vel.x','vel.y','steering_angle','body_angle','yaw_rate','drift_angle'],       help='List of a targets')
    parser.add_argument('--features_list',   nargs="+",  default=['time', 'speed'],                   help='List of features')
    parser.add_argument('--commands_list', nargs="+", default=['cmd.throttle', 'cmd.brake', 'cmd.steering'],          help='List of commands')
    parser.add_argument('--targets_list', nargs="+", default=['speed'],                               help='List of targets')

    my_args = parser.parse_args()
    return my_args
