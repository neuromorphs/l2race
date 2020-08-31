from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pysindy as ps

from pygame.math import Vector2

from random import randrange
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'SINDy':
            from modeling.SINDy.sindy import SINDy
            return SINDy
        return super().find_class(module, name)

class Data:
    """
    A class for simplified loading of simulator data for use with pysindy
    Supports loading data from single or multiple .csv files
    ...

    Attributes
    ----------
    x : np.array or list of np.arrays
        values of features listed in FEATURES variable 
    x_dot : np.array or list of np.arrays
        precalculated derivatives of features listed in DERIVATIVES variable
    u : np.array or list of np.arrays
        values of car commands specified in COMMANDS variable
    t : np.array or list of np.arrays
        timestamps
    multiple_trajectories: bool
        specifies whether data is loaded from single or multiple files
        if True, data is stored as lists of np.arrays
    """

    def __init__(self, data_dir):
        """
        Parameters
        ----------
        data_dir: str
            directory containing .csv files
        """

        full_path = Path.cwd().joinpath(data_dir)
        file_list = [x for x in full_path.glob('*.csv')]

        if len(file_list) == 0:
            raise Exception("Data directory is empty!")

        elif len(file_list) == 1:
            multiple_trajectories = False

            data = pd.read_csv(file_list[0], comment='#')
            data = data.drop_duplicates('time')
            
            t = data['time'].values
            u = data[COMMANDS].values
            x = data[FEATURES].values
            x_dot = data[PRECALCULATED_DERIVATIVES].values

        else:
            multiple_trajectories = True

            t = []
            x = []
            x_dot = []
            u = []
            for f in file_list:
                data = pd.read_csv(f, comment='#')
                data = data.drop_duplicates('time') # removes "duplicate" timestamps 

                t_single = data['time']
                u_single = data[COMMANDS]
                x_single = data[FEATURES]
                x_dot_single = data[PRECALCULATED_DERIVATIVES]

                t.append(t_single.values)
                x.append(x_single.values)
                x_dot.append(x_dot_single.values)
                u.append(u_single.values) 

        self.x = x
        self.x_dot = x_dot
        self.u = u
        self.t = t
        self.multiple_trajectories = multiple_trajectories

class SINDy(ps.SINDy):
    """
    Custom SINDy model
    """
    def __init__(self, optimizer=None, feature_library=None, differentiation_method=None, feature_names=None, t_default=1, discrete_time=False, n_jobs=1):
        super().__init__(optimizer, feature_library, differentiation_method, feature_names, t_default, discrete_time, n_jobs)

    def fit(self, data, calculate_derivatives=True):
        """
        constructs SINDy model
        ...

        Parameters
        ----------
        data: Data object
            training data object constructed using Data class
        calculate_derivatives: bool
            if False, precalculated derivative values will be used.
            Otherwise, they will be calculated during training (differentiation method can be specified)

        Returns
        -------
        model: SINDy model object
            trained SINDy model
        """

        if calculate_derivatives:
            super().fit(x=data.x, u=data.u, t=data.t,
                multiple_trajectories=data.multiple_trajectories)       
        else:
            super().fit(x=data.x, x_dot=data.x_dot, u=data.u, t=data.t,
                multiple_trajectories=data.multiple_trajectories)
        

    def plot(self, test_data, save=False):
        """
        plots trained model for evaluation
        ...

        Parameters
        ----------
        model: SINDy model object
            trained SINDy model
        test_data: Data object
            test data object constructed using Data class
            if test_data contains multiple trajectories, one will be randomly selected for testing
        save: bool
            if set to True, the plot will be saved as .png

        Plots
        -----
        car path (x,y) (ground truth vs. model simulation)
        each feature specified in FEATURES with respect to time (ground truth vs. model simulation)
        """

        if test_data.multiple_trajectories:
            i = randrange(len(test_data.x))
            test_data.x = test_data.x[i]
            test_data.x_dot = test_data.x_dot[i]
            test_data.u = test_data.u[i]
            test_data.t = test_data.t[i]

        u_interp = interp1d(test_data.t, test_data.u, axis=0, kind='cubic',fill_value="extrapolate")
        x_predicted = model.simulate(test_data.x[0], test_data.t, u=u_interp)

        fig, axs = plt.subplots(len(FEATURES)+1, 1, figsize=(9, 16))

        axs[0].plot(test_data.x[:,0], test_data.x[:,1], 'g+', label='simulation (ground truth)')
        axs[0].plot(x_predicted[:,0], x_predicted[:, 1], 'r+', label='model')
        axs[0].invert_yaxis()
        axs[0].legend()
        axs[0].set(title='Car path', xlabel=r'$x$', ylabel=r'$y$')

        for i, feature in enumerate(FEATURES):
            axs[i+1].plot(test_data.t, test_data.x[:,i], 'k', label='true simulation')
            axs[i+1].plot(test_data.t, x_predicted[:,i], 'r--', label='model simulation')
            axs[i+1].legend()
            axs[i+1].set(title=feature, xlabel=r'$t\ [s]$', ylabel=feature)

        plt.tight_layout()
        plt.show()

        if save: plt.savefig('plot.png')

    def save(self, path):
        with open(Path.cwd().joinpath(path), 'wb') as f:
            pickle.dump(self, f)

    def simulate_step(self, car_state, car_command, t, dt):
        u = lambda t: np.array([car_command.steering, car_command.throttle]) # u has to be callable in order to work with pysindy in continuous time
        s0 = np.concatenate([car_state.position_m, [car_state.body_angle_deg]]) # starting state
        
        sim = super().simulate(s0, [t-dt, t], u)

        new_state = dict()
        new_state['position_m'] = Vector2(sim[1,0], sim[1,1])
        new_state['body_angle_deg'] = sim[1,2]

        return new_state

def load_model(path):
    with open(Path.cwd().joinpath(path), 'rb') as f:
        return CustomUnpickler(f).load()

if __name__=='__main__':

    # parameters
    train_dir = 'data' # name of directory relative to l2race root folder
    test_dir = 'data_test'

    FEATURES = ['pos.x', 'pos.y', 'body_angle']
    COMMANDS = ['cmd.throttle', 'cmd.steering']
    PRECALCULATED_DERIVATIVES = []

    # usage example
    optimizer=ps.SR3(threshold=0.01, thresholder='l1', normalize=True, max_iter=1000) # define optimizer
    feature_lib=ps.PolynomialLibrary(degree=2) # define feature library

    model = SINDy(
        optimizer,
        feature_library=feature_lib,
        feature_names=FEATURES+COMMANDS) # define model object

    train_data = Data(train_dir)
    model.fit(train_data)
    model.print()

    test_data = Data(test_dir)
    model.plot(test_data)

    model.save('modeling/SINDy/SINDy_model.pkl')

    #model = load('modeling/SINDy/SINDy_model.pkl')