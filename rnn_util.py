from src.car_state import car_state
from src.globals import *
from src.car import car
import glob
from modeling.sindy import Data


class Dataset(data.Dataset):
    """
    This is a Dataset class providing a proper data format for Pytorch applications
    It inherits from the standard Pytorch dataset class
    """

    def __init__(self,data_dir, args, train=True):
       
        """Initialization"""
             """
        Parameters
        ----------
        data_dir: str
            directory containing .csv files
        """
        # # Take data set of different size for training and testing of RNN
        # if train:
        #     self.exp_len = args.exp_len_train
        # else:
        #     self.exp_len = args.exp_len_test
        data = Data(data_dir)

        # # Recalculate simulation time step from milliseconds to seconds
        # self.dt = args.dt / 1000.0  # s
        # self.epoch_len = args.epoch_len

    def __len__(self):
        """
        Total number of samples.
        In this implementation it is meaningless however needs to be preserved due to Pytorch requirements
        """

        return int(self.epoch_len)

    def __getitem__(self, idx):
        
        """
        When called this function reads the recording file in a form suitable to be used to train/test RNN
        (inputs list to RNN in features array and expected outputs from RNN )
        """

        FEATURES = ['pos.x', 'pos.y','vel.x','vel.y','steering_angle','body_angle','yaw_rate','drift_angle']
        COMMANDS = ['cmd.throttle', 'cmd.steering', 'cmd.brake', 'cmd.reverse']

        data = Data(data_dir)
        # "features" is the array of inputs to the RNN, it consists of states of the car and control input
        # "targets" is the array of car states one time step ahead of "features" at the same index.
        # "targets[i]" is what we expect our network to predict given features[i]
        # file_length: length of the selected recording file
        file_length = data.x[idx].shape[0]

        states = data.x[idx][:(file_length-1),:]
        control = data.u[idx][1:,:]
        target = data.x[idx][1:,:]
 
        features = np.hstack((np.array(states), np.array(control)))
        features = torch.from_numpy(features).float()

        targets = np.array(targets)
        targets = torch.from_numpy(targets).float()

        return features, targets