# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""
import torch

import os
import time
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm import tqdm
import random as rnd
import numpy as np

import collections

from memory_profiler import profile
import timeit

from utilis import get_device, Sequence, Dataset, plot_results, load_data
from args_rnn_cartpole import args as my_args


# Check if GPU is available. If yes device='cuda:0' if not device='cpu'
device = get_device()

args = my_args()
print(args.__dict__)

# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network():
    # Start measuring time - to evaluate performance of the training function
    start = timeit.default_timer()

    # Set seeds
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Make folders if not yet exist
    try:
        os.makedirs('save')
        os.makedirs('log')
    except:
        pass

    lr = args.lr  # learning rate
    batch_size = args.batch_size  # Mini-batch size
    num_epochs = args.num_epochs  # Number of epochs to train the network
    num_rnn_layers = 2
    rnn_hid_size_1 = args.h1_size
    rnn_hid_size_2 = args.h2_size
    rnn_type = 'GRU'

    # Save and Log
    str_target_variable = 'cart-pole-based-rnn'
    str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size_1) + 'H1-' + str(rnn_hid_size_2) + 'H2-'
    filename = str_net_arch + str(rnn_type) + '-' + str_target_variable
    pretrain_model_path = str_net_arch + 'GRU'
    logpath = './log/' + filename + '.csv'
    savepath = './save/' + filename + '.pt'

    ########################################################
    # Create Dataset
    ########################################################
    train_file = '../../data/'+'l2race-Marcin-oval_easy-train.csv'
    val_file = '../../data/'+'l2race-Marcin-oval_easy-test.csv'
    train_features, train_dict, train_targets, target_dict,_,actual_dict, mean_train_features, std_train_features, mean_train_targets, std_train_targets = \
        load_data(train_file, args, savepath, save_normalization_parameters=True)
    dev_features,_, dev_targets, _, _, _, _, _, _, _ = \
        load_data(val_file, args, savepath)

    train_set = Dataset(train_features, train_targets, args)
    dev_set = Dataset(dev_features, dev_targets, args)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers = args.num_workers)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=True, num_workers = args.num_workers)

    # Create RNN instance
    net = Sequence(args)

    # If a pretrained model exists load the parameters from disc and into RNN instance
    # Also evaluate the performance of this pretrained network
    # by checking its predictions on a randomly generated CartPole experiment
    try:
        pre_trained_model = torch.load(args.savepathPre, map_location=torch.device('cpu'))
        print("Loading Model: ", args.savepathPre)

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
        # Evaluate the performance of this pretrained network
        # by checking its predictions on a randomly generated CartPole experiment
        plot_results(net, args)

    except:
        print('No pretrained model available')


    # Print parameter count
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        params += sizes
    print('::: # network parameters: ' + str(params))

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)

    # Select Loss Function
    criterion = nn.MSELoss()  # Mean square error loss function

    # Initialize weights and biases
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

    ########################################################
    # Training
    ########################################################
    print("Starting training...")

    # Create dictionary to store training history
    dict_history = {}
    dict_history['epoch'] = []
    dict_history['time'] = []
    dict_history['lr'] = []
    dict_history['train_loss'] = []
    dict_history['dev_loss'] = []
    dict_history['dev_gain'] = []
    dict_history['test_loss'] = []
    dev_gain = 1

    # Save time to evaluate performance of the training function
    start_time = time.time()

    # Maximal window over which we want to train RNN - look below for explanation of training process
    # win_max = (num_epochs - 1) // args.epochs_per_win
    win_max = 0
    # The epoch_saved variable will indicate from which epoch is the last RNN model,
    # which was good enough to be saved
    epoch_saved = -1
    for epoch in range(num_epochs):

        # With consecutive epochs, when our RNN gets better and better
        # we want to have longer and longer prediction window
        # The schema for training is following:
        # (1) we start with net.initialize_sequence(batch)
        #       in this operation we run RNN in "open loop",
        #       i.e. the inputs are always ground truth (experiment recording)
        #       and outputs are not plugged back into RNN.
        #       The aim of this operation is to let the internal states of RNN cells adjust to data
        # (2) we close the loop
        #       and let the RNN predict the future CartPole states over next win * args.warm_up_len time steps.
        #       In this operation the output of the RNN is plugged back to its input for the next time step
        #       Only control input is taken at every time step from ground truth
        #       We do it with the forward function with terminate=False
        # (3) we reset the memory of all gradients with  optimizer.zero_grad()
        # (4) we continue making prediction for another args.warm_up_len time steps in closed loop,
        #       again with forward function,
        #       this time we return outputs (terminate=True) and save gradients
        # (5) we calculate loss based on the prediction OF THESE LAST args.warm_up_len time_steps
        #       and accordingly update the weights of RNN model
        # win = (epoch // args.epochs_per_win)  # window number over which to optimize the results
        win  = 0
        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        # Set RNN in training mode
        net = net.train()
        # Define variables accumulating training loss and counting training batchs
        train_loss = 0
        train_batches = 0

        # Iterate training over available batches
        # tqdm() is just a function which displays the progress bar
        # Otherwise the line below is the same as "for batch, labels in train_generator:"
        for batch, labels in tqdm(train_generator):  # Iterate through batches

            # Reset the network (internal states of hidden layers and output history not the weights!)
            net.reset()

            # Further modifying the input and output form to fit RNN requirements
            # If GPU available we send tensors to GPU (cuda)
            if torch.cuda.is_available():
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()

            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()

            # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
            net.initialize_sequence(batch)
            # Predict in closed loop - we are not interested in these results yet
            net(win * args.warm_up_len, batch)

            # Reset memory of gradients
            optimizer.zero_grad()

            # Forward propagation - These are the results from which we calculate the update to RNN weights
            # GRU Input size must be (seq_len, batch, input_size)
            out = net(args.warm_up_len, batch, terminate=True)

            # Get loss
            loss = criterion(out[:, (win + 1) * args.warm_up_len:(win + 2) * args.warm_up_len],
                             labels[:, (win + 1) * args.warm_up_len:(win + 2) * args.warm_up_len])

            # Backward propagation
            loss.backward()

            # Gradient clipping - prevent gradient from exploding
            torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

            # Update parameters
            optimizer.step()

            # Update variables for loss calculation
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss
            train_batches += 1  # Accumulate count so we can calculate mean later

        ###########################################################################################################
        # Validation - Iterate batches
        ###########################################################################################################

        # Set the network in evaluation mode
        net = net.eval()

        # Define variables accumulating evaluation loss and counting evaluation batches
        dev_loss = 0
        dev_batches = 0

        for (batch, labels) in tqdm(dev_generator):

            # Reset the network (internal states of hidden layers and output history not the weights!)
            net.reset()

            # Further modifying the input and output form to fit RNN requirements
            # If GPU available we send tensors to GPU (cuda)
            if torch.cuda.is_available():
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()

            # Convert Pytorch tensors to numpy matrices to inspect them - just for debugging purpose.
            # Variable explorers of IDEs are often not compatible with Pytorch format
            # np_batch = batch.numpy()
            # np_label = labels.numpy()

            # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
            net.initialize_sequence(batch)
            # Forward propagation
            # GRU Input size must be (look_back_len, batch, input_size)
            out = net((win_max + 1) * args.warm_up_len, batch, terminate=True)
            # Get loss
            # For evaluation we always calculate loss over the whole maximal prediction period
            # This allow us to compare RNN models from different epochs
            loss = criterion(out[:, args.warm_up_len:(win_max + 2) * args.warm_up_len],
                             labels[:, args.warm_up_len:(win_max + 2) * args.warm_up_len])

            # Update variables for loss calculation
            batch_loss = loss.detach()
            dev_loss += batch_loss  # Accumulate loss
            dev_batches += 1  # Accumulate count so we can calculate mean later

        # Reset the network (internal states of hidden layers and output history not the weights!)
        net.reset()
        # Get current learning rate
        # TODO: I think now the learning rate do not change during traing, or it is not a right way to get this info.
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        # Write the summary information about the training for the just completed epoch to a dictionary
        dict_history['epoch'].append(epoch)
        dict_history['lr'].append(lr_curr)
        dict_history['train_loss'].append(train_loss.detach().cpu().numpy() / train_batches)
        dict_history['dev_loss'].append(dev_loss.detach().cpu().numpy() / dev_batches)

        # Get relative loss gain for network evaluation
        if epoch >= 1:
            dev_gain = (dict_history['dev_loss'][epoch - 1] - dict_history['dev_loss'][epoch]) / \
                       dict_history['dev_loss'][epoch - 1]
        dict_history['dev_gain'].append(dev_gain)

        # Print the summary information about the training for the just completed epoch
        print('\nEpoch: %3d of %3d | '
              'LR: %1.5f | '
              'Train-L: %6.4f | '
              'Val-L: %6.4f | '
              'Val-Gain: %3.2f |' % (dict_history['epoch'][epoch], num_epochs - 1,
                                     dict_history['lr'][epoch],
                                     dict_history['train_loss'][epoch],
                                     dict_history['dev_loss'][epoch],
                                     dict_history['dev_gain'][epoch] * 100))

        # Save the best model with the lowest dev loss
        # Always save the model from epoch 0
        # TODO: this is a bug: you should only save the model from epoch 0 if there is no pretraind network
        if epoch == 0:
            min_dev_loss = dev_loss
        # If current loss smaller equal than minimal till now achieved loss,
        # save the current RNN model and save its loss as minimal ever achieved
        if dev_loss <= min_dev_loss:
            epoch_saved = epoch
            min_dev_loss = dev_loss
            torch.save(net.state_dict(), args.savepath)
            print('>>> saving best model from epoch {}'.format(epoch))
        else:
            print('>>> We keep model from epoch {}'.format(epoch_saved))
        # Evaluate the performance of the current network
        # by checking its predictions on a randomly generated CartPole experiment
        # plot_results(net, args)

    # When finished the training print the final message
    print("Training Completed...                                               ")
    print(" ")

    # Calculate the total time it took to run the function
    stop = timeit.default_timer()
    total_time = stop - start

    # Return the total time it took to run the function
    return total_time


if __name__ == '__main__':
    time_to_accomplish = train_network()
    print('Total time of training the network: ' + str(time_to_accomplish))
