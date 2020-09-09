"""
L2RACE SINDy framework
Built on PySINDy: https://github.com/dynamicslab/pysindy

@author Ante Maric
"""

from pathlib import Path
from functools import wraps
import pickle
import numpy as np
import pandas as pd
import pysindy as ps
import copy

from pygame.math import Vector2

from src.car_model import car_model
from src.car_state import car_state
from src.car_command import car_command

from random import randrange
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler for loading saved models in different scripts
    """
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
            raise Exception("Directory is empty or non-existent!")

        elif len(file_list) == 1: # TODO generalize single file loading as case of multiple file loading (one item list)
            multiple_trajectories = False

            data = pd.read_csv(file_list[0], comment='#')
            data = data.drop_duplicates('time')  # removes "duplicate" timestamps; TODO check if needed
            
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
                data = data.drop_duplicates('time') # TODO check if needed

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
    def __init__(self, optimizer=None, feature_library=None, differentiation_method=None, features=None, commands=None, t_default=1, discrete_time=False, n_jobs=1):
        super().__init__(optimizer, feature_library, differentiation_method, features+commands, t_default, discrete_time, n_jobs)
        self.features = features
        self.commands = commands

        self.command_attributes = [x.split('.')[1] for x in self.commands] # maps .csv file command names to car_command class attributes

        # maps .csv file state names to car_state class attributes
        self.feature_attributes = []
        all_feature_attributes = ['position_m', 'velocity_m_per_sec','speed_m_per_sec','accel_m_per_sec_2', \
            'steering_angle_deg', 'body_angle_deg','yaw_rate_deg_per_sec', 'drift_angle_deg']

        if 'pos.x' in features or 'pos.y' in features:
            self.feature_attributes.append('position_m')
        if 'vel.x' in features or 'vel.y' in features:
            self.feature_attributes.append('velocity_m_per_sec')
        if 'speed' in features:
            self.feature_attributes.append('speed_m_per_sec')
        if 'accel.x' in features or 'accel.y' in features:
            self.feature_attributes.append('accel_m_per_sec_2')
        
        self.feature_attributes += [x for x in all_feature_attributes if '_'.join(x.split('_')[:2]) in features]


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
                multiple_trajectories=data.multiple_trajectories, unbias=True)       
        else:
            super().fit(x=data.x, x_dot=data.x_dot, u=data.u, t=data.t,
                multiple_trajectories=data.multiple_trajectories, unbias=True)
        

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

        plt.rcParams.update({'font.size': 18})

        if test_data.multiple_trajectories:
            i = randrange(len(test_data.x))
            test_data.x = test_data.x[i]
            test_data.x_dot = test_data.x_dot[i]
            test_data.u = test_data.u[i]
            test_data.t = test_data.t[i]

        u_interp = interp1d(test_data.t, test_data.u, axis=0, kind='cubic',fill_value="extrapolate")

        # use newer solver (multiple available methods):
        #def flipped_arguments(fun):
        #    @wraps(fun)
        #    def fun_flipped(x, y):
        #        return fun(y, x)
        #    return fun_flipped

        #solve_ivp_wrapped = lambda fun, y0, t, *args, **kwargs: solve_ivp(flipped_arguments(fun), tuple([t[0],t[-1]]), \
        #    y0, *args, method='Radau', dense_output=True, t_eval=t, atol=1e-4, rtol=1e-2, **kwargs).y.T

        #x_predicted = model.simulate(test_data.x[0], test_data.t, u=u_interp, integrator=solve_ivp_wrapped)
        
        # ----------------------------------------------------------------------------------------------------------------

        # set different integrator parameters for odeint
        #x_predicted = model.simulate(test_data.x[0], test_data.t, u=u_interp, atol=1e-4, rtol=1e-2, hmin=1e-6, hmax=1e-3)

        x_predicted = model.simulate(test_data.x[0], test_data.t, u=u_interp)

        _, axs = plt.subplots(1, 1, figsize=(13, 8))

        axs.plot(test_data.x[:,0], test_data.x[:,1], 'g+', label='simulation (ground truth)')
        axs.plot(x_predicted[:,0], x_predicted[:, 1], 'r+', label='model')
        axs.invert_yaxis()
        axs.legend()
        axs.set(title='Car path', xlabel=r'$x$', ylabel=r'$y$')
        plt.tight_layout()

        if save: plt.savefig('car_path.png')

        _, axs = plt.subplots(len(FEATURES), 1, figsize=(12, 18))

        for i, feature in enumerate(FEATURES):
            axs[i].plot(test_data.t, test_data.x[:,i], 'k', label='true simulation')
            axs[i].plot(test_data.t, x_predicted[:,i], 'r--', label='model simulation')
            axs[i].legend()
            axs[i].set(title=feature, xlabel=r'$t\ [s]$', ylabel=feature)

        plt.tight_layout()

        if save: plt.savefig('features.png')

        plt.show()

    def save(self, path):
        with open(Path.cwd().joinpath(path), 'wb') as f:
            pickle.dump(self, f)

    def simulate_step(self, curr_state, curr_command, t, dt):
        cmds = [getattr(curr_command, x) for x in self.command_attributes] # get values of all used commands
        u = lambda t: np.array(cmds) # u has to be callable in order to work with pysindy in continuous time

        states  = [getattr(curr_state, x) for x in self.feature_attributes] # get values of all used states    
        s0 = np.concatenate([s if hasattr(s, '__iter__') else [s] for s in states]) # stitch them into a starting state

        sim = super().simulate(s0, [t-dt, t], u)

        new_state = copy.deepcopy(curr_state) # copy state object

        # construct new state
        i = 0
        for f in self.feature_attributes: # TODO solve case when feature of just one coordinate is used (e.g., vel.y, but not vel.x) ; do we need this?
            if f in ['position_m', 'velocity_m_per_sec','accel_m_per_sec_2']:
                setattr(new_state, f, Vector2(sim[1,i], sim[1,i+1]))
                i += 2
            else:
                setattr(new_state, f, sim[1,i])
                i += 1

        return new_state

def load_model(path):
    with open(Path.cwd().joinpath(path), 'rb') as f:
        return CustomUnpickler(f).load()

if __name__=='__main__':

    # parameters
    train_dir = 'data' # name of directory relative to l2race root folder
    test_dir = 'data_test'

    # respect the feature order from .csv files; in case of position, velocity, acceleration, both coordinates need to be used
    FEATURES = ['pos.x', 'pos.y', 'vel.x', 'vel.y', 'body_angle']
    COMMANDS = ['cmd.throttle', 'cmd.steering', 'cmd.brake']
    PRECALCULATED_DERIVATIVES = [] # TODO check use with precalculated derivatives

    # usage example
    optimizer=ps.SR3(threshold=0.01, thresholder='l1', normalize=True, max_iter=100000) # TODO test different optimizers
    feature_lib=ps.PolynomialLibrary(degree=2) # TODO test different feature libs

    #smoothed_fd = ps.SmoothedFiniteDifference(drop_endpoints=True)

    model = SINDy(
        optimizer,
        feature_library=feature_lib,
        features=FEATURES,
        #differentiation_method=smoothed_fd,
        commands=COMMANDS) # TODO test different differetiation methods

    train_data = Data(train_dir)
    model.fit(train_data)
    model.print()

    test_data = Data(test_dir)
    model.plot(test_data, save=True)

    model.save('modeling/SINDy/SINDy_model.pkl')

    #model = load('modeling/SINDy/SINDy_model.pkl')