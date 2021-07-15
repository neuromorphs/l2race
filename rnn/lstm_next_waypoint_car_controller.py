# driver controller
import logging
from src.car import car
from src.car_command import car_command
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import math
import time

# import required to process csv files with pandas
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from pandas import concat

M_PER_PIXEL = 0.1

INPUT_DIM = 32
DECODER_INPUT_DIM = 4
OUTPUT_DIM = 4
HID_DIM = 32 
N_LAYERS = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

logger = logging.getLogger(__name__)

WB = 2.9  # [m] wheel base of vehicle 
LFC = 3.0 # [m] look ahead distance
K = 0.1 # look forward gain

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

#       input = self.dropout(src)
        input = src

        outputs, (hidden, cell) = self.rnn(input)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(1)

#       input = self.dropout(input)

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(1))

        #prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        #first input to the decoder
        input = trg[:,0,:]

        for t in range(1, trg_len):

            #insert input, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            output = torch.squeeze(output, 1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:,t,:] if teacher_force else output #top1

        return outputs


class lstm_next_waypoint_car_controller:
    """
    car controller: gets state and provides control signal.
    This reference implementation is a basic PID controller that aims for the next waypoint.

    """
    def __init__(self, my_car: car = None):
        """
        :type my_car: car object
        :param car: All car info: car_state and track
        For the controller the user needs to know information about the current car state and its position in the track
        Then, the user should implement a controller that generate a new car command to apply to the car
        """
        self.car = my_car
        self.car_command = car_command()

        # internet values kp=0.1 ki=0.001 kd=2.8
        self.steer_Kp = 0.8
        self.steer_Ki = 0.000500
        self.steer_Kd = 4.8

        self.steer_p_error = 0
        self.steer_i_error = 0
        self.steer_d_error = 0

        self.max_speed = 9 

        self.T1 = 85    # input sentence length  # time steps of car states+env
        self.T2 = 10    # output sentence length # time steps of car commands

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # import the multi feature csv
        multifeature_csv = pd.read_csv(r'/home/arthurlobo/l2race/data/l2race-Wolfie_oval_easy_pure_pursuit+manual_09082020.csv', header=None)

        # diplay the contents of the csv file with NO processing
        self.L2Race_Data_processed = multifeature_csv.iloc[:,].values

        self.nrows = self.L2Race_Data_processed.shape[0]
        # process the data, take all data except the first column
        self.L2Race_Data_car_commands = multifeature_csv.iloc[0:self.nrows, 2:6].values
        self.L2Race_Data_car_state = multifeature_csv.iloc[0:self.nrows, 6:38].values
        print(self.L2Race_Data_car_commands.shape, self.L2Race_Data_car_state.shape)
        self.before_scaling_car_state = self.L2Race_Data_car_state
        self.before_scaling_car_commands = self.L2Race_Data_car_commands
        #print(self.before_scaling.shape)

        # normalize features
        self.scaler_car_state = MinMaxScaler(feature_range=(0, 1))
        self.scaled_car_state = self.scaler_car_state.fit_transform(self.before_scaling_car_state)
        self.scaler_car_commands = MinMaxScaler(feature_range=(0, 1))
        self.scaled_car_commands = self.scaler_car_commands.fit_transform(self.before_scaling_car_commands)
        print(self.scaled_car_state.shape)
        print(self.scaled_car_commands.shape)
        self.L2Race_Data_car_state = self.scaled_car_state        # X
        self.L2Race_Data_car_commands = self.scaled_car_commands  # y


        self.enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        self.dec = Decoder(OUTPUT_DIM, DECODER_INPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        self.model = Seq2Seq(self.enc, self.dec, self.device).to(self.device)

        # Load seq2seq model for car command prediction
        self.model.load_state_dict(torch.load('/home/arthurlobo/l2race/seq2seq/test/l2race-model_09092020_oval_easy_4.pt'))
        self.X = torch.zeros(self.T1,32).to(self.device)
        self.y = torch.zeros(self.T2,4).to(self.device)           # command input to decoder is 0, LSTM should evolve from car state input and fed back output (predicted command)
        self.i = 0
        self.car_state_env = torch.zeros(1,32).to(self.device)
        self.car_cmd = np.zeros((1,32))
        self.do_pure_pursuit = 0



    def read(self, cmd:car_command):

        self.car_command = cmd
        '''computes the control and returns it as a standard keyboard/joystick command'''
        waypoint_distance = self.car.track.get_distance_to_nearest_segment(car_state=self.car.car_state,
                                                                           x_car=self.car.car_state.position_m.x,
                                                                           y_car=self.car.car_state.position_m.y)
        self.car_state_env[0,0] = waypoint_distance
        next_waypoint_id = self.car.track.get_nearest_waypoint_idx(car_state=self.car.car_state,
                                                                   x=self.car.car_state.position_m.x,
                                                                   y=self.car.car_state.position_m.y)

        w_ind = next_waypoint_id
        wp1_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp1_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,1] = wp1_x
        self.car_state_env[0,2] = wp1_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp2_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp2_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,3] = wp2_x
        self.car_state_env[0,4] = wp2_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp3_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp3_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,5] = wp3_x
        self.car_state_env[0,6] = wp3_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp4_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp4_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,7] = wp4_x
        self.car_state_env[0,8] = wp4_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp5_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp5_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,9] =  wp5_x
        self.car_state_env[0,10] = wp5_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp6_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp6_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,11] = wp6_x
        self.car_state_env[0,12] = wp6_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp7_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp7_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,13] = wp7_x
        self.car_state_env[0,14] = wp7_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp8_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp8_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,15] = wp8_x
        self.car_state_env[0,16] = wp8_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp9_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp9_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,17] = wp9_x
        self.car_state_env[0,18] = wp9_y

        w_ind = w_ind+1
        if w_ind >= len(self.car.track.waypoints_x):
          w_ind = 0
        wp10_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
        wp10_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
        self.car_state_env[0,19] = wp10_x
        self.car_state_env[0,20] = wp10_y


        self.car_state_env[0,21] = self.car.car_state.position_m.x
        self.car_state_env[0,22] = self.car.car_state.position_m.y
        self.car_state_env[0,23] = self.car.car_state.velocity_m_per_sec.x
        self.car_state_env[0,24] = self.car.car_state.velocity_m_per_sec.y
        self.car_state_env[0,25] = self.car.car_state.speed_m_per_sec
        self.car_state_env[0,26] = self.car.car_state.accel_m_per_sec_2.x
        self.car_state_env[0,27] = self.car.car_state.accel_m_per_sec_2.y
        self.car_state_env[0,28] = self.car.car_state.steering_angle_deg
        self.car_state_env[0,29] = self.car.car_state.body_angle_deg
        self.car_state_env[0,30] = self.car.car_state.yaw_rate_deg_per_sec
        self.car_state_env[0,31] = self.car.car_state.drift_angle_deg

        #For every time-step a new state+env vector is received, perform scaling before input to LSTM
        self.X = self.X.squeeze(0)
#        print(self.X.shape)
        if (self.i < self.T1):
          self.scaled1 = self.scaler_car_state.transform(self.car_state_env.cpu())
          self.X[self.T1-self.i-1,:] = torch.from_numpy(self.scaled1).float().to(self.device)
          self.i = self.i + 1
        else:
#          print("----------------------------------------------------------")
          self.X = torch.roll(self.X,1,0)
          self.scaled1 = self.scaler_car_state.transform(self.car_state_env.cpu())
          self.X[0,:] = torch.from_numpy(self.scaled1).float().to(self.device)


        #print(self.X.shape,self.y.shape)          # torch.Size([85, 32]) torch.Size([10, 4])
        self.X = self.X.unsqueeze(0)               # Make X and y 3D tensor for input to model
        self.y1 = self.y.unsqueeze(0)
#        print("3", self.X.shape, self.y1.shape)



        self.yhat = self.model(self.X, self.y1, 0)



#       print(self.yhat.shape)        [10, 1, 4]
        self.yhat = self.yhat.cpu()
        self.yhat = self.yhat.squeeze(1)
        self.yhat = self.yhat.detach().numpy()
        #print(self.yhat.shape)                         # (10,4)
        # invert scaling for prediction
        self.y_hat_inv = self.scaler_car_commands.inverse_transform(self.yhat)
#        print("predicted commands\n")
#        print(self.y_hat_inv)


        if (self.do_pure_pursuit == 1):

           w_ind = next_waypoint_id
           wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
           wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
           car_x = self.car.car_state.position_m.x 
           car_y = self.car.car_state.position_m.y
           yaw_angle = self.car.car_state.body_angle_deg%360   #  degrees, increases CW (on screen!) with zero pointing to right/east
           yaw_angle_rad = (yaw_angle * math.pi)/180.0
           rear_x = (car_x - (WB/2.0) * math.cos(yaw_angle_rad))
           rear_y = (car_y - (WB/2.0) * math.sin(yaw_angle_rad))

           v = self.car.car_state.speed_m_per_sec  
           Lf = K * v + LFC                   # update lookahead distance

           distance = math.sqrt((wp_x - car_x)**2 + (wp_y - car_y)**2)
#           print("waypoint index = ", w_ind, "distance = ", distance, "Lf =", Lf)
           while (distance < Lf):
#             w_ind -= 1                                               # += 1 for clockwise track 
#             if w_ind > -1:                                           # < len(self.car.track.waypoints_x):  for clockwise track
             w_ind += 1
             if w_ind  < len(self.car.track.waypoints_x):
                wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
                wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
             else:
#                w_ind = len(self.car.track.waypoints_x) - 1           # 0  for clockwise track 
                w_ind = 0
                wp_x = self.car.track.waypoints_x[w_ind] * M_PER_PIXEL
                wp_y = self.car.track.waypoints_y[w_ind] * M_PER_PIXEL
             car_x = self.car.car_state.position_m.x 
             car_y = self.car.car_state.position_m.y
             distance = math.sqrt((wp_x - car_x)**2 + (wp_y - car_y)**2)

           alpha = math.atan2(wp_y - rear_y, wp_x - rear_x) - yaw_angle_rad
           steering_angle = math.atan2(2.0 * WB * math.sin(alpha)/Lf, 1.0)
             


        # T2 car command tuples are output by LSTM decoder for past T1 car state tuples
#        print("4",self.y_hat_inv.shape)
        self.car_cmd = self.y_hat_inv[1,:]
        self.car_command.steering = np.clip(self.car_cmd[0], -1.0, 1.0) #/3.0)
        self.car_command.throttle = self.car_cmd[1]
        self.car_command.brake = self.car_cmd[2]
        self.car_command.reverse = int(abs(self.car_cmd[3]))
#       print(self.yhat.shape)
        self.y = torch.from_numpy(self.yhat).to(self.device)


#        self.__update_steering_error(cte=waypoint_distance)
#        self.car_command.throttle = 0.2 if self.car.car_state.speed_m_per_sec < self.max_speed else 0
#        self.car_command.steering = self.output_steering_angle()
#        self.car_command.steering = steering_angle
        return self.car_command

    def __update_steering_error(self, cte):
        pre_cte = self.steer_p_error

        self.steer_p_error = cte
        self.steer_i_error += cte
        self.steer_d_error = cte - pre_cte

    def output_steering_angle(self):
        angle = - self.steer_Kp * self.steer_p_error - self.steer_Ki * self.steer_i_error - \
               self.steer_Kd * self.steer_d_error
        return angle
