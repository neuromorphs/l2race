# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin

The file generates an CartPole experiments and loads pretrained RNN network. It feeds
"""

import torch
import torch.utils.data.dataloader

import collections

from utilis import get_device, plot_results, create_rnn_instance
# Parameters of RNN
from args_rnn_cartpole import args as my_args

# Check if GPU is available. If yes device='cuda:0' if not device='cpu'
device = get_device()

# Get arguments as default or from terminal line
args = my_args()
# Print the arguments
print(args.__dict__)


def test_network():

    """
    This function create RNN instance based on parameters saved on disc and also creates the CartPole instance.
    The actual work of evaluation prediction results is done in plot_results function
    """

    # Network architecture:
    rnn_name = args.rnn_name
    inputs_list = args.inputs_list
    outputs_list = args.outputs_list

    load_rnn = args.load_rnn  # If specified this is the name of pretrained RNN which should be loaded
    path_save = args.path_save

    # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, rnn_name, inputs_list, outputs_list\
        = create_rnn_instance(rnn_name, inputs_list, outputs_list, load_rnn, path_save, device)

    plot_results(net=net, args=args, filepath='../../data/oval_easy_12_rounds.csv', warm_up_len=1, seq_len=600, comment='Testing the RNN',
                 inputs_list=inputs_list, outputs_list=outputs_list, save=True, closed_loop_enabled=True)


if __name__ == '__main__':
    test_network()
