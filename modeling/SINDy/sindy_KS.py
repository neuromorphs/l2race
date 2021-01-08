"""
L2RACE SINDy framework
Built on PySINDy: https://github.com/dynamicslab/pysindy

@author Ante Maric
"""

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pysindy as ps
from functools import wraps

from scipy.interpolate import interp1d

from random import randrange
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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

                data['body_angle_rad'] = np.deg2rad(data['body_angle_deg'])
                data['steering_angle_rad'] = np.deg2rad(data['steering_angle_deg'])

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


def solver_simulate(model_speed, model_steering, model_body_angle, test_data):
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

    print("Simulating ...")

    i = randrange(0, len(test_data.t))
    test_data.x = test_data.x[i]
    test_data.pos = test_data.pos[i]
    test_data.x_dot = test_data.x_dot[i]
    test_data.u = test_data.u[i]
    test_data.t = test_data.t[i]

    # use newer solver (multiple available methods):
    # def flipped_arguments(fun):
    #     @wraps(fun)
    #     def fun_flipped(x, y):
    #         return fun(y, x)
    #     return fun_flipped

    # def solve_ivp_wrapped(fun, y0, t, *args, **kwargs):
    #     return solve_ivp(flipped_arguments(fun), tuple([t[0], t[-1]]),
    #                      y0, *args, method='Radau', dense_output=True,
    #                      t_eval=t, atol=1e-6, rtol=1e-4, **kwargs).y.T

    u = lambda t: np.array(cmds)

    s0 = test_data.x[0]
    pos0 = test_data.pos[0]
    res = np.array([np.hstack((pos0, s0))])

    for i in range(1, test_data.t.shape[0]):
        t = test_data.t[i]
        dt = t - test_data.t[i-1]
        cmds = test_data.u[i-1, 0:2]
        sim_speed = model_speed.simulate([s0[0]], [t-dt, t], u)

        cmds = test_data.u[i-1, 2]
        sim_steering = model_steering.simulate([s0[1]], [t-dt, t], u)

        cmds = res[-1, 2:4]
        sim_body_angle = model_body_angle.simulate([s0[2]], [t-dt, t], u)

        pos0[0] += sim_speed[1]*np.cos(sim_body_angle[1])*dt  # x
        pos0[1] += sim_speed[1]*np.sin(sim_body_angle[1])*dt  # y
        s0 = np.hstack((sim_speed[1], sim_steering[1], sim_body_angle[1]))
        res = np.append(res, [np.hstack((pos0, s0))], axis=0)

    return test_data, res


def plot(test_data, simulated_data, save=False):
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

    print("Plotting ...")

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
        axs[i].set(xlabel=r'$t\ [s]$', ylabel=feature)

    plt.tight_layout()

    if save:
        plt.savefig('features.png')

    plt.show()

if __name__ == '__main__':

    # NOTE: in this script, seperate models are created for each of the state variables
    # this is the easiest way to get the satisfactory coefficients, although a bit unrealistic for real case scenarios
    # for a single-model implementation, take a look at past versions of this script on github
    # results are similar and can likely be improved for both implementations

    train_dir = 'Train'
    test_dir = 'Test'

    # STEERING_MAX = 0.5  # can be checked in simulator

    # FIT SPEED MODEL
    FEATURES = ['speed_m_per_sec']
    COMMANDS = ['command.throttle', 'command.brake']
    PRECALCULATED_DERIVATIVES = []

    library_functions = [lambda x: x]
    library_function_names = [lambda x: x]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
    )

    model_speed = SINDy(
        features=FEATURES,
        commands=COMMANDS,
        feature_library=custom_library,
        optimizer=ps.STLSQ(threshold=2.5)
        )

    train_data = Data(train_dir)
    model_speed.fit(train_data)
    model_speed.print()

    # FIT STEERING MODEL
    FEATURES = ['steering_angle_rad']
    COMMANDS = ['command.steering']
    PRECALCULATED_DERIVATIVES = []

    # library_functions = [
    #     lambda x, y: (np.abs(STEERING_MAX*y-x) > np.deg2rad(1)) * np.sign(STEERING_MAX*y-x),
    # ]

    # library_function_names = [
    #     lambda x, y: '(|{}*y-x| > radians(1)) * sign({} * '.format(STEERING_MAX, STEERING_MAX) + y + ' - ' + x + ')',
    # ]

    # custom_library = ps.CustomLibrary(
    #     library_functions=library_functions,
    #     function_names=library_function_names,
    # )

    model_steering = SINDy(
        features=FEATURES,
        commands=COMMANDS,
        feature_library=ps.PolynomialLibrary(degree=1),  # custom_library
        optimizer=ps.STLSQ(threshold=2)
        )

    train_data = Data(train_dir)
    model_steering.fit(train_data)
    model_steering.print()

    # FIT BODY ANGLE MODEL
    FEATURES = ['body_angle_rad']
    COMMANDS = ['speed_m_per_sec', 'steering_angle_rad']
    PRECALCULATED_DERIVATIVES = []

    library_functions = [
        lambda x, y: x*np.tan(y)
    ]

    library_function_names = [
        lambda x, y: x + '*' + 'tan(' + y + ')'
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
    )

    model_body_angle = SINDy(
        features=FEATURES,
        commands=COMMANDS,
        feature_library=custom_library,
        optimizer=ps.STLSQ()
        )

    train_data = Data(train_dir)
    model_body_angle.fit(train_data)
    model_body_angle.print()

    # PLOT ENTIRE MODEL ON TEST DATA
    FEATURES = ['speed_m_per_sec', 'steering_angle_rad', 'body_angle_rad']
    COMMANDS = ['command.throttle', 'command.brake', 'command.steering']
    test_data = Data(test_dir)
    test_data, simulated_data = solver_simulate(model_speed,
                                                model_steering,
                                                model_body_angle,
                                                test_data)
    plot(test_data, simulated_data)

    # coefficients
    # speed_coef = model_speed.optimizer.coef_
    # steering_coef = model_steering.optimizer.coef_
    # body_angle_coef = model_body_angle.optimizer.coef_

    # SAVE MODEL TO CSV
    print('Saving model ...')
    precision = 3
    with open('modeling/SINDy/KS_model.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(model_speed.equations(precision))
        writer.writerow(model_steering.equations(precision))
        writer.writerow(model_body_angle.equations(precision))
