"""
L2RACE SINDy framework
Built on PySINDy: https://github.com/dynamicslab/pysindy

@author Ante Maric
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pysindy as ps
# from functools import wraps

from random import randrange
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp


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
        else:
            self.multiple_trajectories = True
            self.t = []
            self.x = []
            self.x_dot = []
            self.u = []
            self.pos = []
            for f in file_list:
                data = pd.read_csv(f, comment='#')

                data = data.drop_duplicates('time')

                data['body_angle_deg'] = np.deg2rad(data['body_angle_deg'])
                data['steering_angle_deg'] = np.deg2rad(data['steering_angle_deg'])

                t_single = data['time']
                u_single = data[COMMANDS]
                x_single = data[FEATURES]
                pos = data[['position_m.x', 'position_m.y']]
                x_dot_single = data[PRECALCULATED_DERIVATIVES]

                self.t.append(t_single.values)
                self.x.append(x_single.values)
                self.u.append(u_single.values)
                self.pos.append(pos.values)
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
                        unbias=True)
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

        axs.plot(test_data.pos[:, 0], test_data.pos[:, 1], 'g+',
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
            axs[i].plot(test_data.t, simulated_data[:, i+2], 'r--',
                        label='model')
            axs[i].legend()
            axs[i].set(title=feature, xlabel=r'$t\ [s]$', ylabel=feature)

        plt.tight_layout()

        if save:
            plt.savefig('features.png')

        plt.show()


def solver_simulate(model, test_data):
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

    i = randrange(len(test_data.t))
    test_data.x = test_data.x[i]
    test_data.pos = test_data.pos[i]
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
    pos0 = test_data.pos[0]
    res = np.array([np.hstack((pos0, s0))])

    for i in range(1, test_data.t.shape[0]):
        t = test_data.t[i]
        dt = t - test_data.t[i-1]
        cmds = test_data.u[i-1]
        sim = model.simulate(s0, [t-dt, t], u)

        # --------------------------------------------------------------------
        # newer solver
        # sim = model.simulate(s0, [t-dt, t], u, integrator=solve_ivp_wrapped)
        # --------------------------------------------------------------------

        pos0[0] += s0[0]*np.cos(s0[2])*dt  # x
        pos0[1] += s0[0]*np.sin(s0[2])*dt  # y
        s0 = sim[1, :]
        res = np.append(res, [np.hstack((pos0, s0))], axis=0)

    return test_data, res


def euler_simulate(model, test_data):
    """
    simulates test data from beginning to end using euler stepping

    !!! WARNING: model is currently hard-coded in this function
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
    t = test_data.t[0]
    u = test_data.u[0]
    pos = test_data.pos[0][0]
    x = test_data.x[0][0]
    res = np.array([np.hstack((pos, x))])

    for i in range(1, t.shape[0]):
        dt = t[i]-t[i-1]

        d_speed = 2.473*u[i-1, 0] * dt  # * 2.81/2.473

        d_steer = 1.552 * (np.abs((0.5*u[i-1, 1]-res[i-1, 3])) > np.deg2rad(1)) \
                        * np.sign(0.5 * u[i-1, 1] - res[i-1, 3]) * dt  # * 2/1.552

        d_angle = 0.383 * res[i-1, 2] * np.tan(res[i-1, 3]) * dt  # * 0.38776/0.383
        d_x = res[i-1, 2] * np.cos(res[i-1, 4]) * dt
        d_y = res[i-1, 2] * np.sin(res[i-1, 4]) * dt

        new = np.array([res[i-1, 0] + d_x,
                        res[i-1, 1] + d_y,
                        res[i-1, 2] + d_speed,
                        res[i-1, 3] + d_steer,
                        res[i-1, 4] + d_angle])

        res = np.vstack((res, new))

    test_data.x = test_data.x[0]
    test_data.pos = test_data.pos[0]
    test_data.x_dot = test_data.x_dot[0]
    test_data.u = test_data.u[0]
    test_data.t = test_data.t[0]

    return test_data, res


if __name__ == '__main__':

    train_dir = 'Train'
    test_dir = 'Test'

    FEATURES = [
                'speed_m_per_sec',
                'steering_angle_deg',
                'body_angle_deg'
                ]

    COMMANDS = [
                'command.throttle',
                'command.steering',
                # 'command.brake'
                ]

    PRECALCULATED_DERIVATIVES = []

    library_functions = [
        lambda x: x,
        lambda x, y: (np.abs(0.5*y-x) > np.deg2rad(1)) * np.sign(0.5*y-x),
        lambda x, y: x*np.tan(y)
    ]

    library_function_names = [
        lambda x: x,
        lambda x, y: '(|0.5*y-x| > radians(1)) * sign(0.5 * ' + y + ' - ' + x + ')',
        lambda x, y: x + '*' + 'tan(' + y + ')'
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
    )

    model = SINDy(
        features=FEATURES,
        commands=COMMANDS,
        feature_library=custom_library,
        optimizer=ps.SR3(threshold=0.18, nu=0.0003, tol=1e-5, thresholder='l1',
                         fit_intercept=False, normalize=False, max_iter=100000),
        differentiation_method=ps.FiniteDifference(order=1)
        )

    train_data = Data(train_dir)
    model.fit(train_data)
    model.print()

    test_data = Data(test_dir)

    test_data, sim = euler_simulate(model, test_data)
    # test_data, sim = solver_simulate(model, test_data)

    model.plot(test_data, sim, save=False)
