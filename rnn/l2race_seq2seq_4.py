# L2RACE seq2seq LSTM Encoder-Decoder NN multi-variate time series prediction
# Adapted from Sequence to Sequence Learning with Neural Networks by Ben Trevett
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# Example usage: python l2race_seq2seq_3.py 85 10 1
# Make sure to specify the proper path for the .csv data file
# Author: Arthur Lobo  8/31/2020
#                      9/1/2020  Changed normalization code, added inverse scaling, test code (uses validation set), third argument (1=train, 0=test)
#                      9/6/2020  Car commands and Car state+env scaled separately, changed variable name space to state for all occurences
#                      9/7/2020  Flip car state+env tensor in time dimension before input to LSTM decoder

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
from sklearn.model_selection import train_test_split
from pandas import concat

SEED = 5678 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

T1 = int(sys.argv[1])    # input sentence length  # time steps of car states+env
T2 = int(sys.argv[2])    # output sentence length # time steps of car commands

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the multi feature csv
multifeature_csv = pd.read_csv(r'../../data/l2race_oval_easy.csv', header=None) 
 
# diplay the contents of the csv file with NO processing
L2Race_Data_processed = multifeature_csv.iloc[:,].values
#print (L2Race_Data_processed.shape)

nrows = L2Race_Data_processed.shape[0]
# process the data, take all data except the first column
L2Race_Data_car_commands = multifeature_csv.iloc[0:nrows, 2:6].values
L2Race_Data_car_state = multifeature_csv.iloc[0:nrows, 6:38].values
print(L2Race_Data_car_commands.shape, L2Race_Data_car_state.shape)
before_scaling_car_state = L2Race_Data_car_state
before_scaling_car_commands = L2Race_Data_car_commands
#print(before_scaling.shape)

# normalize features
scaler_car_state = MinMaxScaler(feature_range=(0, 1))
scaled_car_state = scaler_car_state.fit_transform(before_scaling_car_state)
scaler_car_commands = MinMaxScaler(feature_range=(0, 1))
scaled_car_commands = scaler_car_commands.fit_transform(before_scaling_car_commands)
#print(scaled)
print(scaled_car_state.shape)
print(scaled_car_commands.shape)
#L2Race_Data_car_commands = scaled[:,0:4]       # y 
L2Race_Data_car_state = scaled_car_state        # X
L2Race_Data_car_commands = scaled_car_commands  # y

#print("y_unscaled\n", before_scaling_car_commands[T1:T1+T2,:])

#print("car commands")
#print (L2Race_Data_car_commands)
#print("car state")
#print (L2Race_Data_car_state)


#print (L2Race_Data_processed.shape[0]) # x-axis
#print (L2Race_Data_processed.shape[1]) # y-axis
#print (L2Race_Data_car_commands.shape) # target
#print (L2Race_Data_car_state.shape)    # input

X_train1, X_val1, Y_train1, Y_val1 = train_test_split(L2Race_Data_car_state, L2Race_Data_car_commands, test_size=0.1, shuffle=False)
X_train = X_train1
Y_train = Y_train1
X_val = X_val1
Y_val = Y_val1
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)  # (39593, 32) (4400, 32) (39593, 4) (4400, 4)

X_train_sentences = np.empty([X_train.shape[0]-T1-T2,T1,32])
for i in range(0,X_train.shape[0]-T1-T2):
  data1 = np.flip(X_train[i:i+T1],axis=0).copy()
  X_train_sentences[i] = data1

#print(X_train_sentences)
#print("X_train", X_train_sentences.shape)

X_val_sentences = np.empty([X_val.shape[0]-T1-T2,T1,32])
for i in range(0,X_val.shape[0]-T1-T2):
  data1 = np.flip(X_val[i:i+T1],axis=0).copy() 
  X_val_sentences[i] = data1

#print(X_val_sentences)
#print("X_val", X_val_sentences.shape)

Y_train_sentences = np.empty([Y_train.shape[0]-T1-T2,T2,4])
for i in range(0,Y_train.shape[0]-T1-T2):
  data1 = np.flip(Y_train[T1+i:T1+i+T2],axis=0).copy() 
  Y_train_sentences[i] = data1

#print(Y_train_sentences)
#print("Y_train", Y_train_sentences.shape)

Y_val_sentences = np.empty([Y_val.shape[0]-T1-T2,T2,4])
for i in range(0,Y_val.shape[0]-T1-T2):
  data1 = np.flip(Y_val[T1+i:T1+i+T2],axis=0).copy() 
  Y_val_sentences[i] = data1

#print(Y_val_sentences)
#print("Y_val", Y_val_sentences.shape)

X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train_sentences, X_val_sentences)]
Y_train, Y_val = [torch.tensor(arr, dtype=torch.float32) for arr in (Y_train_sentences, Y_val_sentences)]

#print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape) # torch.Size([4916, 85, 32]) torch.Size([4916, 10, 4]) torch.Size([462, 85, 32]) torch.Size([462, 10, 4])

train_ds = TensorDataset(X_train, Y_train)
val_ds = TensorDataset(X_val, Y_val)

batch_size = 128
jobs = 4

#DataLoader returns a batch of shape [batch_size, seq_len, features]
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=jobs)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=jobs)

print("Loaded train and val data")

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


INPUT_DIM = 32 
DECODER_INPUT_DIM = 4
OUTPUT_DIM = 4 
HID_DIM = 32 
N_LAYERS = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DECODER_INPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.MSELoss() 

def train(model, train_dataloader, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    len_train_dataloader = len(train_dataloader.dataset)
    dl_size = int(len_train_dataloader/batch_size)
    
    for idx, (src, trg) in enumerate(train_dataloader):
        
      if idx < dl_size:
#       print('BatchIdx {}, src.shape {}, trg.shape {}'.format(idx, src.shape, trg.shape))
        # permute the dimensions to match [seq_len, batch_size, features] or just use batch_first=True in the LSTM
        src = src.to(device='cuda')
        trg = trg.to(device='cuda')
        
        optimizer.zero_grad()
        
        # input should have shape (seq_len, batch, input_size)

        output = model(src, trg)
        output = output.transpose(1,0)
        
        #trg = [trg len, batch size]          # torch.Size([128, 3, 4])
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
       
        output = output.reshape(1,batch_size*T2*4)
        trg = trg.reshape(1,batch_size*T2*4)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_dataloader)

def evaluate(model, val_dataloader, criterion):
    
    model.eval()
    
    epoch_loss = 0

    len_val_dataloader = len(val_dataloader.dataset) 
    dl_size = int(len_val_dataloader/batch_size)
    
    with torch.no_grad():
    
        for idx, (src, trg) in enumerate(val_dataloader):
          
#         print('BatchIdx {}, src.shape {}, trg.shape {}'.format(idx, src.shape, trg.shape))
          if idx < dl_size:
            src = src.to(device='cuda')
            trg = trg.to(device='cuda')

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output = output.transpose(1,0)
            output_dim = output.shape[-1]
            
            output = output.reshape(1,batch_size*T2*4)
            trg = trg.reshape(1,batch_size*T2*4)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(val_dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1000 
CLIP = 1

best_valid_loss = float('inf')

train_flag = int(sys.argv[3])

if (train_flag == 1):

  f = open("loss_11_flip.txt", "w")

#  model.load_state_dict(torch.load('l2race-model_09072020_2_flip.pt'))

  for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_dl, optimizer, criterion, CLIP)

    valid_loss = evaluate(model, val_dl, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'l2race-model_09072020_2_flip.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.7f}')
    print(f'\t Val. Loss: {valid_loss:.7f}')
    f.write("{:d} {:.7f} {:.7f}\n".format(epoch, train_loss, valid_loss))
    f.flush()

  f.close()

# Testing 
model.load_state_dict(torch.load('l2race-model_09072020_2_flip.pt'))
test_loss = evaluate(model, val_dl, criterion)
print(f'Test Loss: {test_loss:.7f}')

#print(X_val.shape, Y_val.shape)   # torch.Size([462, 85, 32]) torch.Size([462, 10, 4])

offset = 0
#X = X_val[0,:,:].to(device)        # select an arbitrary car state history for input
X_in = X_val1[offset:offset+T1,:]
X_in = np.flip(X_in, axis=0).copy()
X = torch.from_numpy(X_in).to(device)  
# alternative first command input for seq2seq decoder 
#y = np.zeros((T2,4))
#for i in range(0,T2):
# y[i,0] = 0.0  # steering
# y[i,1] = 0.5  # throttle
# y[i,2] = 0.0  # brake
# y[i,3] = 0.0  # reverse
#y = torch.from_numpy(scaler_car_commands.transform(y)).to(device)
#y = y.float()
y = torch.zeros(T2, 4).to(device)  # command input to decoder is 0, LSTM should evolve from car state input and fed back output (predicted command)
#print(X.shape,y.shape)            # torch.Size([85, 32]) torch.Size([10, 4])
#X = X.to(device)
X = X.float()
X = X.unsqueeze(0)
y = y.unsqueeze(0)
print("2",X.shape,y.shape)

yhat = model(X, y, 0)

#print(yhat)
#print(yhat.shape, X.shape, y.shape)
yhat = yhat.cpu()
yhat = yhat.squeeze(1)
yhat = yhat.detach().numpy()
print("yhat.shape=", yhat.shape)              # (10,4)
test_y = Y_val1[offset+T1:offset+T1+T2,:]
test_y = np.flip(test_y, axis=0).copy()
print("test_y.shape=", test_y.shape)          # (10,4)
# invert scaling for prediction
y_hat_inv = scaler_car_commands.inverse_transform(yhat)
# invert scaling for actual
y_inv = scaler_car_commands.inverse_transform(test_y)
print("1", y_hat_inv.shape, y_inv.shape)   # (10,4) (10,4)
print("y_hat_inv\n", y_hat_inv)
print("y_inv\n", y_inv)
print(Y_train.shape[0])
#print("y_unscaled\n", before_scaling_car_commands[Y_train.shape[0]+offset+T1+95:Y_train.shape[0]+offset+T1+T2+95,:])
rmse = math.sqrt(mean_squared_error(y_hat_inv, y_inv))
print(f'Test RMSE: {rmse:.7f}')
