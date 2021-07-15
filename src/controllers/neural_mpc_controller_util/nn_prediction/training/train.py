import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from src.controllers.neural_mpc_controller_util.neural_mpc_settings import *
from src.controllers.neural_mpc_controller_util.nn_prediction.training.util import *
import json


nn_prediction_folder_path = "src/controllers/neural_mpc_controller_util/"


def train_network():


    nn_settings ={
        "predict_delta" : PREDICT_DELTA,
        "normalize_data": NORMALITE_DATA,
        "model_name": MODEL_NAME,
        "cut_invariants": CUT_INVARIANTS,
        "epochs": NUMBER_OF_EPOCHS,
        "batch_size": BATCH_SIZE
    }
 

    # load the dataset
    #time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
    train_data = np.loadtxt(nn_prediction_folder_path + '/nn_prediction/training/data/{}'.format(TRAINING_DATA_FILE), delimiter=',')
    print("train_data.shape", train_data.shape)

    # limit data for debugging
    # train_data = np.loadtxt('nn_prediction/training_data_small.csv', delimiter=',')


    # split into input (X) and output (y) variables
    #x1,x2,x3,x4,x5,x6,x7,u1,u2,
    x = train_data[:,1:10]
    #x1n,x2n,x3n,x4n,x5n,x6n,x7n
    y = train_data[:,10:]


    #delta is the difference between state and next_state
    delta =  y[:] - x[:,:7]

    # Cut position away from input data
    x = x[:,2:]
    #if we want to train the network on the state changes instead of the state, use this
    if PREDICT_DELTA:
        y = delta

    # Augmentation for lots of lots of data
    # x, y = augment_data(x,y)

    # Normalize data
    scaler_x = preprocessing.MinMaxScaler().fit(x)
    scaler_y = preprocessing.MinMaxScaler().fit(y)

    if(NORMALITE_DATA):
        x = scaler_x.transform(x)
        y = scaler_y.transform(y)


    # keras model
    model = Sequential()
    model.add(Dense(128, input_dim=7, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(7))


    # compile
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit
    history = model.fit(x, y, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1)

    # Save model and normalization constants
    model_path =  nn_prediction_folder_path + 'nn_prediction/models/{}'.format(MODEL_NAME)
    scaler_x_path = nn_prediction_folder_path + 'nn_prediction/models/{}/scaler_x.pkl'.format(MODEL_NAME)
    scaler_y_path = nn_prediction_folder_path + 'nn_prediction/models/{}/scaler_y.pkl'.format(MODEL_NAME)
    nn_settings_path =  nn_prediction_folder_path + 'nn_prediction/models/{}/nn_settings.json'.format(MODEL_NAME)

    model.save(model_path)
    joblib.dump(scaler_x, scaler_x_path) 
    joblib.dump(scaler_y, scaler_y_path) 
    with open(nn_settings_path, "w") as outfile:
        outfile.write(json.dumps(nn_settings))


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(nn_prediction_folder_path + 'nn_prediction/models/{}/accuracy_curve'.format(MODEL_NAME))

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(nn_prediction_folder_path + 'nn_prediction/models/{}/loss_curve'.format(MODEL_NAME))


    #Evaluate
    _, accuracy = model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))






if __name__ == '__main__':

    train_network()
