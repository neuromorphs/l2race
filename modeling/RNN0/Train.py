# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""

import os
import timeit

# Various
import torch.optim as optim
import torch.utils.data.dataloader
from torch.optim import lr_scheduler
from tqdm import tqdm

import numpy as np

from memory_profiler import profile


# Custom functions
from utilis import *
# Parameters of RNN
from args_rnn_cartpole import args as my_args


# Check if GPU is available. If yes device='cuda:0' if not device='cpu'
device = get_device()

args = my_args()
print(args.__dict__)


# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network(load_pretrained):
    # Start measuring time - to evaluate performance of the training function
    start = timeit.default_timer()

    # Set seeds
    set_seed(args)

    # Make folders if not yet exist
    try:
        os.makedirs('save')
    except FileExistsError:
        pass

    # Save relevant arguments from args and set hardcoded arguments
    lr = args.lr  # learning rate
    batch_size = args.batch_size  # Mini-batch size
    num_epochs = args.num_epochs  # Number of epochs to train the network
    
    rnn_name = args.rnn_name
    rnn_full_name = rnn_name[:4] + str(len(args.inputs_list))+'IN-' + rnn_name[4:] + '-'+str(len(args.outputs_list))+'OUT'

    # Create log for this file
    net_index = 0
    while True:

        csv_path = args.path_save + rnn_full_name + '-' + str(net_index) + '.csv'
        name_pretrained_default = rnn_full_name + '-' + str(net_index-1)
        if os.path.isfile(csv_path):
            pass
        else:
            rnn_full_name += '-' + str(net_index)
            f = open(csv_path, 'w')
            f.write('RNN NAME: ' + rnn_name + '\n')
            f.write('RNN FULL NAME: ' + rnn_full_name + '\n')
            f.write('INPUTS: ' + ', '.join(map(str, args.inputs_list)) + '\n')
            f.write('OUTPUTS: ' + ', '.join(map(str, args.outputs_list)) + '\n')
            f.close()
            break

        net_index += 1

    ########################################################
    # Create Dataset
    ########################################################

    train_features, train_targets = load_data(args.train_file_name,args)
    dev_features, dev_targets = load_data(args.val_file_name,args)

    train_set = Dataset(train_features, train_targets, args)
    dev_set = Dataset(dev_features, dev_targets, args)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=True, num_workers=args.num_workers)

    # Create RNN instance
    net = Sequence(rnn_name=rnn_name, inputs_list=args.inputs_list, outputs_list=args.outputs_list)

    # If a pretrained model exists load the parameters from disc and into RNN instance
    if load_pretrained:
        try:
            load_pretrained_rnn(net, args.path_pretrained)
            # Evaluate the performance of this pretrained network
            # plot_results(net, args)
        except FileNotFoundError:
            print('No pretrained model found')
            load_pretrained = False

    # Print parameter count
    # print_parameter_count(net) # Seems not to function well

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)

    #TODO: Verify if scheduler is working. Try tweaking parameters of below scheduler and try cyclic lr scheduler
    
    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Select Loss Function
    criterion = nn.MSELoss()  # Mean square error loss function

    # Initialize weights and biases - should be only applied if no pretrained net loaded
    if not load_pretrained:
        initialize_weights_and_biases(net)

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


    # The epoch_saved variable will indicate from which epoch is the last RNN model,
    # which was good enough to be saved
    epoch_saved = -1
    for epoch in range(num_epochs):

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
            net.initialize_sequence(rnn_input=batch,
                                    warm_up_len=args.warm_up_len,
                                    all_input=False,
                                    stack_output=False)

            # Reset memory of gradients
            optimizer.zero_grad()

            # Forward propagation - These are the results from which we calculate the update to RNN weights
            # GRU Input size must be (seq_len, batch, input_size)
            out = net.initialize_sequence(rnn_input=batch[:args.warm_up_len, :, :],
                                          warm_up_len=args.warm_up_len,
                                          all_input=False,
                                          stack_output=True)

            # Get loss
            loss = criterion(out[:, args.warm_up_len: 2 * args.warm_up_len],
                             labels[:, args.warm_up_len: 2 * args.warm_up_len])

            # Backward propagation
            loss.backward()

            # Gradient clipping - prevent gradient from exploding
            torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

            # Update parameters
            optimizer.step()
            scheduler.step()
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
            net.initialize_sequence(rnn_input=batch,
                                    warm_up_len=args.warm_up_len,
                                    all_input=False,
                                    stack_output=False)
            # Forward propagation
            # GRU Input size must be (look_back_len, batch, input_size)
            out = net.initialize_sequence(rnn_input=batch[args.warm_up_len:, :, :],
                                          warm_up_len=args.warm_up_len,
                                          all_input=False,
                                          stack_output=True)
            # Get loss
            # For evaluation we always calculate loss over the whole maximal prediction period
            # This allow us to compare RNN models from different epochs
            loss = criterion(out[:, args.warm_up_len: 2 * args.warm_up_len],
                             labels[:, args.warm_up_len: 2 * args.warm_up_len])

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
            torch.save(net.state_dict(), args.path_save + rnn_full_name + '.pt')
            print('>>> saving best model from epoch {}'.format(epoch))
        else:
            print('>>> We keep model from epoch {}'.format(epoch_saved))
        # Evaluate the performance of the current network
        # by checking its predictions on a randomly generated CartPole experiment
        # plot_results(net, args, val_file)

    # When finished the training print the final message
    print("Training Completed...                                               ")
    print(" ")

    # Calculate the total time it took to run the function
    stop = timeit.default_timer()
    total_time = stop - start

    # Return the total time it took to run the function
    return total_time


if __name__ == '__main__':
    time_to_accomplish = train_network(load_pretrained=False)
    print('Total time of training the network: ' + str(time_to_accomplish))
