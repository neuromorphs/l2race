"""
L2RACE SINDy framework
Built on PySINDy: https://github.com/dynamicslab/pysindy

@author Ante Maric
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pysindy as ps
import copy
# from functools import wraps
from sklearn import preprocessing
#from sklearn.linear_model import MultiTaskElasticNetCV, ElasticNet
#from sklearn.linear_model import OrthogonalMatchingPursuit

from pygame.math import Vector2
from random import randrange
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp


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

    def __init__(self, data_dir, normalize=False, start=0, skip_n=-1):
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
        else:
            self.multiple_trajectories = True
            self.t = []
            self.x = []
            self.x_dot = []
            self.u = []
            for f in file_list:
                data = pd.read_csv(f, comment='#')

                # data preprocessing
                # TODO why is this needed? jitter? precision?
                data = data.drop_duplicates('time')
                if skip_n > 0:
                    data = data.iloc[start::skip_n]

                t_single = data['time']
                u_single = data[COMMANDS]
                x_single = data[FEATURES]
                x_dot_single = data[PRECALCULATED_DERIVATIVES]

                self.t.append(t_single.values)
                if normalize:
                    self.x.append(preprocessing.normalize(x_single.values, norm='max'))
                    self.u.append(preprocessing.normalize(u_single.values, norm='max'))
                else:
                    self.x.append(x_single.values)
                    self.u.append(u_single.values)
                self.x_dot.append(x_dot_single.values)


class SINDy(ps.SINDy):
    """
    Custom SINDy model
    """
    def __init__(self, optimizer=None, feature_library=None,
                 differentiation_method=None, features=None, commands=None,
                 t_default=1, discrete_time=False, n_jobs=1):
        super().__init__(optimizer, feature_library, differentiation_method,
                         features+commands, t_default, discrete_time, n_jobs)
        self.features = features
        self.commands = commands

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
            Otherwise, they will be calculated during training
            (differentiation method can be specified)

        Returns
        -------
        model: SINDy model object
            trained SINDy model
        """

        if calculate_derivatives:
            super().fit(x=data.x, u=data.u, t=data.t,
                        multiple_trajectories=data.multiple_trajectories,
                        unbias=True)  # TODO what does unbias exactly do?
        else:
            super().fit(x=data.x, x_dot=data.x_dot, u=data.u, t=data.t,
                        multiple_trajectories=data.multiple_trajectories,
                        unbias=True)

    def plot(self, test_data, simulated_data, save=False):
        """
        plots trained model for evaluation
        ...

        Parameters
        ----------
        model: SINDy model object
            trained SINDy model
        test_data: Data object
            test data object constructed using Data class
            if test_data contains multiple trajectories, one will be randomly
            selected for testing
        save: bool
            if set to True, the plot will be saved as .png

        Plots
        -----
        car path (x,y) (ground truth vs. model simulation)
        each feature specified in FEATURES with respect to time
        (ground truth vs. model simulation)
        """

        plt.rcParams.update({'font.size': 18})

        _, axs = plt.subplots(1, 1, figsize=(13, 8))

        axs.plot(test_data.x[:, 0], test_data.x[:, 1], 'g+',
                 label='true path')
        axs.plot(simulated_data[:, 0], simulated_data[:, 1], 'r+',
                 label='model')
        axs.invert_yaxis()
        axs.legend()
        axs.set(title='Car path', xlabel=r'$x$', ylabel=r'$y$')
        plt.tight_layout()

        if save:
            plt.savefig('car_path.png')

        _, axs = plt.subplots(len(FEATURES), 1, figsize=(12, 18))

        for i, feature in enumerate(FEATURES):
            axs[i].plot(test_data.t, test_data.x[:, i], 'k',
                        label='true data')
            axs[i].plot(test_data.t, simulated_data[:, i], 'r--',
                        label='model')
            axs[i].legend()
            axs[i].set(title=feature, xlabel=r'$t\ [s]$', ylabel=feature)

        plt.tight_layout()

        if save:
            plt.savefig('features.png')

        plt.show()

    def save(self, path):
        with open(Path.cwd().joinpath(path), 'wb') as f:
            pickle.dump(self, f)

    def simulate_step(self, curr_state, curr_command, t, dt):
        """
        simulates one step forward from starting state and returns new state
        (to be used with the ghost car feature)
        ...

        Parameters
        ----------
        curr_state: src.car_state.car_state object
            current (starting) state of the car
        curr_command: src.car_state.car_command object
            command passed to the car at starting time
        t: np.float64
            next timestep
        dt: np.float64
            next-current timestep difference

        Returns
        -------
        new_state: src.car_state.car_state object
            simulated state at next timestep
        """

        # get values of all used commands
        cmds = [getattr(curr_command, x) for x in self.commands]

        def u(t):  # u has to be callable in order to work with pysindy
            return np.array(cmds)

        # get values of all used states
        states = [getattr(curr_state, x) for x in self.features]
        # stitch them into a starting state
        s0 = np.concatenate([s if hasattr(s, '__iter__')
                             else [s] for s in states])

        sim = super().simulate(s0, [t-dt, t], u)

        new_state = copy.copy(curr_state)

        # construct new state
        # solve case when feature of just one coordinate is used ?
        # (e.g., vel.y, but not vel.x) ; probably not needed
        i = 0
        for f in self.features:
            if f in ['position_m', 'velocity_m_per_sec', 'accel_m_per_sec_2']:
                setattr(new_state, f, Vector2(sim[1, i], sim[1, i+1]))
                i += 2
            else:
                setattr(new_state, f, sim[1, i])
                i += 1

        return new_state


def simulate_step_by_step(model, test_data):
    """
    simulates test data from beginning to end using trained model
    simulation is done "step by step" - each timestep is simulated separately
    using the result of the previous timestep as the starting state
    ...

    Parameters
    ----------
    model: SINDy model object
        trained SINDy model
    test_data: Data object
        test data object constructed using Data class
        if test_data contains multiple trajectories, one will be randomly
        selected for testing

    Returns
    -------
    test_data: np.array() (temporary)
        test data that was randomly chosen from test_dir
    res: np.array()
        resulting simulated data
    """

    # load random file from Test folder TODO iterate through all ?
    i = randrange(len(test_data.t))
    test_data.x = test_data.x[i]
    test_data.x_dot = test_data.x_dot[i]
    test_data.u = test_data.u[i]
    test_data.t = test_data.t[i]

    # use newer solver (multiple available methods):
    # def flipped_arguments(fun):
    #    @wraps(fun)
    #    def fun_flipped(x, y):
    #        return fun(y, x)
    #    return fun_flipped

    # def solve_ivp_wrapped(fun, y0, t, *args, **kwargs):
    #    return solve_ivp(flipped_arguments(fun), tuple([t[0], t[-1]]),
    #                     y0, *args, method='Radau', dense_output=True,
    #                     t_eval=t, atol=1e-4, rtol=1e-2, **kwargs).y.T

    def u(t):
        return np.array(cmds)

    s0 = test_data.x[0]
    res = np.array([s0])

    for i in range(1, test_data.t.shape[0]):
        t = test_data.t[i]
        dt = t - test_data.t[i-1]
        cmds = test_data.u[i-1]
        sim = model.simulate(s0, [t-dt, t], u)

        # sim = model.simulate(s0, [t-dt, t], u, integrator=solve_ivp_wrapped)

        # ---------------------------------------------------------------------

        # set different integrator parameters for odeint
        # sim = model.simulate(s0, [t-dt, t], u,
        #                      atol=1e-4, rtol=1e-2,
        #                      hmin=1e-6, hmax=1e-3)

        s0 = sim[1, :]
        res = np.append(res, [s0], axis=0)

    return test_data, res


def load_model(path):
    with open(Path.cwd().joinpath(path), 'rb') as f:
        return CustomUnpickler(f).load()


if __name__ == '__main__':

    # parameters
    train_dir = 'Train'  # name of directory relative to l2race root folder
    test_dir = 'Test'

    # respect the feature order from .csv files
    # in case of pos, vel, accel, both coordinates need to be used
    FEATURES = ['position_m.x', 'position_m.y',
                # 'velocity_m_per_sec.x', 'velocity_m_per_sec.y',
                'speed_m_per_sec',
                # 'accel_m_per_sec_2.x', 'accel_m_per_sec_2.y',
                'steering_angle_deg',
                # 'body_angle_deg',
                'yaw_rate_deg_per_sec',
                'drift_angle_deg',
                'body_angle_sin', 'body_angle_cos'
                ]

    COMMANDS = ['command.throttle', 'command.steering', 'command.brake']

    # TODO check use with precalculated derivatives
    PRECALCULATED_DERIVATIVES = []

    # custom library construction
    # TODO solve numerical issues with division
    library_functions = [
        lambda x: x,
        lambda x, y: x*y,
        lambda x, y: x**2 * y,
        lambda x: 1./x,  # (x+1e-21),
        lambda x, y: x/y,  # (y+1e-21),
        lambda x, y, z: x*y/z,  # (z+1e-21),
        lambda x, y, z: x*np.sin(y+z),  # TODO try math.sin
        lambda x, y, z: x*np.cos(y+z)  # TODO try math.cos
        # lambda x, y, z: (1-abs(2 * x/y)) * z  # friciton steering constraint
                                                # maybe not needed ?
    ]
    library_function_names = [
        lambda x: x,
        lambda x, y: x + '*' + y,
        lambda x, y: x + '^2 *' + y,
        lambda x: '1/' + x,
        lambda x, y: x + '/' + y,
        lambda x, y, z: '(' + x + '*' + y + ')/' + z,
        lambda x, y, z: x + '*' + 'sin(' + y + '+' + z + ')',
        lambda x, y, z: x + '*' + 'cos(' + y + '+' + z + ')'
        # lambda x, y, z: 'friction_steering_constraint(' + x + ',' + y + ',' + z + ')'
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names
    )

    # TODO use ps.IdentityLibrary to ommit position terms ?

    model = SINDy(
        features=FEATURES,
        commands=COMMANDS,

        # feature_library=ps.PolynomialLibrary(degree=3),
        # feature_library=ps.FourierLibrary(n_frequencies=3),
        # feature_library=ps.PolynomialLibrary(degree=3) + custom_library,
        feature_library=custom_library,

        # TODO cross-validate for threshold parameter
        optimizer=ps.SR3(threshold=0.012, thresholder='l1', max_iter=100000),
        # optimizer=ElasticNet(l1_ratio=0.9, fit_intercept=False,
        #                      max_iter=10000, selection='random'),
        # optimizer=MultiTaskElasticNetCV(l1_ratio=0.9, fit_intercept=False,
        #                                max_iter=1000, selection='random', verbose=2),
        # optimizer=ps.STLSQ(threshold=0.001, alpha=0.03, max_iter=10000),
        # optimizer=OrthogonalMatchingPursuit(n_nonzero_coefs=8,
        #                                    fit_intercept=False),

        differentiation_method=ps.FiniteDifference(order=1)
        )

    train_data = Data(train_dir, normalize=False, start=100, skip_n=100)
    model.fit(train_data)
    model.print()

    test_data = Data(test_dir, normalize=False)
    test_data, sim = simulate_step_by_step(model, test_data)

    model.plot(test_data, sim)

    model.save('modeling/SINDy/SINDy_model.pkl')