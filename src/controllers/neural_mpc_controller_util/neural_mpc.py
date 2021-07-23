import logging
import sys
from collections import deque

import matplotlib

from l2race_utils import my_logger, reload_class_if_modified

sys.path.insert(0, './commonroad-vehicle-models/PYTHON/')

from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from src.controllers.neural_mpc_controller_util.neural_mpc_settings import *
from src.controllers.neural_mpc_controller_util.util import *

import numpy as np
from scipy.integrate import odeint
import shapely.geometry as geom
import matplotlib.pyplot as plt

import time
import math

logger = my_logger(__name__)

'''
Can calculate the next control step due to the cost function
'''
from src.controllers.car_controller import car_controller
from src.car import car
from src.track import track

class CarController(car_controller):
    """
    Implementation of a neural MPPI MPC controller
    """

    def __init__(self, car: car, track: track, predictor="euler", model_name=None):
        """
        Initilalize the MPPI MPC car controller
        @param track{Track}: The track which the car should complete 
        @param predictor{enum:("euler"|"odeint"|"nn") }: The type of prediction the controller uses 
        @param model_name {String} (optional): Required if prediction = "nn", defines the name of the neural network that solves the trajectory prediction 
        @return: None
        """

        # Global
        super().__init__(car)
        self.g = neural_mpc_settings()

        # Racing objects
        self.car_state = [1, 1, 1, 1, 1, 1, 1]
        self.track = track  # Track waypoints for drawing

        # Control with common-road models
        self.predictior = predictor
        # self.parameters = parameters_vehicle1()
        self.parameters = parameters_vehicle2()
        self.tEulerStep = self.g.T_EULER_STEP  # [s] One step of solving the ODEINT or EULER
        self.g = neural_mpc_settings()

        # Control with neural network
        if predictor == "nn":
            print("Setting up controller with neural network")
            if model_name is not None:
                from src.controllers.neural_mpc_controller_util.nn_prediction.prediction import NeuralNetworkPredictor
                self.nn_predictor = NeuralNetworkPredictor(model_name=model_name)

        # MPPI data
        self.simulated_history = []  # Hostory of simulated car states
        self.simulated_costs = []  # Hostory of simulated car states
        self.last_control_input = [0, 0]
        self.best_control_sequenct = []

        # Data collection
        self.collect_distance_costs = []
        self.collect_acceleration_costs = []
        self.counter=0
        # Get track ready
        # self.update_trackline()
        self.stats=self.mpc_stats(self, size=3000)

    def set_state(self, car_state, track):
        """
        Overwrite the controller's car_state
        Maps the repetitive variables of the car state onto the trained space
        Fot the L2Race deployment the states have to be transformed into [m] and [deg] instead of [pixel] and [rad]
        @param state{array<float>[7]}: the current state of the car
        """

        pos_x = car_state.position_m[0]
        pos_y = car_state.position_m[1]
        speed = car_state.speed_m_per_sec
        steering_angle_rad = car_state.steering_angle_deg / 360 * 2 * 3.1415
        body_angle_rad = car_state.body_angle_deg / 360 * 2 * 3.1415
        yaw_rate_deg_rad_sec = car_state.yaw_rate_deg_per_sec / 360 * 2 * 3.1415
        drift_angle_rad = car_state.drift_angle_deg / 360 * 2 * 3.1415

        self.car_state = [pos_x, pos_y, steering_angle_rad, speed, body_angle_rad, yaw_rate_deg_rad_sec,
                          drift_angle_rad]

        # Map repetitive states
        self.car_state[4] = self.car_state[4] % 6.28
        if self.car_state[4] > 3.14:
            self.car_state[4] = self.car_state[4] - 6.28
        self.track = track

    def update_trackline(self):
        """
        Update the the next points of the trackline due to the current car state
        Those waypoints are used for the cost function and drawing the simulated history
        """

        # save only Next NUMBER_OF_NEXT_WAYPOINTS points of the track
        waypoint_modulus = self.track.waypoints.copy()
        waypoint_modulus.extend(waypoint_modulus[:self.g.NUMBER_OF_NEXT_WAYPOINTS])

        closest_to_car_position = self.track.get_nearest_waypoint_idx(x=self.car_state[0], y=self.car_state[1])
        first_waypoint = closest_to_car_position + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        last_waypoint = (
                closest_to_car_position
                + self.g.NUMBER_OF_NEXT_WAYPOINTS
                + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        )

        waypoint_modulus = waypoint_modulus[first_waypoint:last_waypoint]

        self.trackline = geom.LineString(waypoint_modulus)

    def car_dynamics(self, x, t, u, p):
        """
        Dynamics of the simulated car from common road
        To use other car dynamics than the defailt ones, comment out here
        @param x: The cat's state
        @param t: array of times where the state has to be evaluated
        @param u: Control input that is applied on the car
        @param p: The car's physical parameters
        @returns: the commonroad car dynamics function, that can be integrated for calculating the state evolution
        """
        # f = vehicle_dynamics_ks(x, u, p)
        f = vehicle_dynamics_st(x, u, p)
        # f = vehicle_dynamics_std(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        return f

    def simulate_step(self, state, control_input):
        """
        Calculate the next system state due to a given state and control input
        for one time step
        @state{array<float>[7]}: the system's state
        @control_input{array<float>[7]}: the applied control input 
        returns: the simulated car state {array<float>[7]} after T_EULER_STEP [s]
        """
        t = np.arange(0, self.g.T_EULER_STEP, self.tEulerStep)

        if (self.predictior == "euler"):
            x_next = solveEuler(self.car_dynamics, state, t, args=(control_input, self.parameters))[-1]
        elif (self.predictior == "odeint"):
            x_next = odeint(self.car_dynamics, state, t, args=(control_input, self.parameters))[-1]
        else:
            x_next = self.nn_predictor.predict_next_state(state, control_input)[:, ]

        # Convert yaw angle to trained space -pi, pi
        x_next[4] = x_next[4] % 6.28
        if x_next[4] > 3.14:
            x_next[4] = x_next[4] - 6.28

        return x_next

    def simulate_trajectory(self, control_inputs):
        """
        Simulates a hypothetical trajectory of the car due to a list of control inputs
        @control_inputs: list<control_input> The list of apllied control inputs over time
        returns: simulated_trajectory{list<state>}: The simulated trajectory due to the given control inputs, cost{float}: the cost of the whole trajectory
        """

        simulated_state = self.car_state
        simulated_trajectory = []
        cost = 0
        index = 0

        for control_input in control_inputs:
            if cost > self.g.MAX_COST:
                cost = self.g.MAX_COST
                # continue

            simulated_state = self.simulate_step(simulated_state, control_input)

            simulated_trajectory.append(simulated_state)

            index += 1

        cost = self.cost_function(simulated_trajectory)
        return simulated_trajectory, cost

    def simulate_trajectory_distribution(self, control_inputs_distrubution):
        """
        Simulate and rage a distribution of hypothetical trajectories of the car due to multiple control sequences
        @control_inputs_distrubution: list<control sequence> A distribution of control sequences
        returns: results{list<state_evolutions>}: The simulated trajectories due to the given control sequences, costs{array<float>}: the cost for each trajectory
        """

        # if we predict the trajectory distribution with a neural network, we have to swap the axes for speedup.
        if self.predictior == "nn":
            return self.simulate_trajectory_distribution_nn(control_inputs_distrubution)

        self.simulated_history = []
        results = []
        costs = np.zeros(len(control_inputs_distrubution))

        i = 0
        for control_input in control_inputs_distrubution:
            simulated_trajectory, cost = self.simulate_trajectory(control_input)

            results.append(simulated_trajectory)
            costs[i] = cost
            i += 1

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    def simulate_trajectory_distribution_nn(self, control_inputs_distrubution):

        """
        Simulate and rage a distribution of hypothetical trajectories of the car due to multiple control sequences
        This method does the same like simulate_trajectory_distribution but it is optimized for a fast use for neural networks. 
        Since the keras model can process much data at once, we have to swap axes to parallelly compute the rollouts.
        @control_inputs_distrubution: list<control sequence> A distribution of control sequences
        returns: results{list<state_evolutions>}: The simulated trajectories due to the given control sequences, costs{array<float>}: the cost for each trajectory
        """

        control_inputs_distrubution = np.swapaxes(control_inputs_distrubution, 0, 1)
        results = []
        states = np.array(len(control_inputs_distrubution[0]) * [self.car_state])
        for control_inputs in control_inputs_distrubution:
            states = self.nn_predictor.predict_multiple_states(states, control_inputs)
            results.append(states)

        results = np.array(results)
        results = np.swapaxes(results, 0, 1)

        costs = []
        for result in results:
            cost = self.cost_function(result)
            costs.append(cost)

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    def static_control_inputs(self):

        """
        Sample primitive hand crafted control sequences
        This method was only for demonstration purposes and is no longer used.
        @returns: results{list<control sequence>}: A primitive distribution of control sequences
        """
        control_inputs = [
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, 0]],  # No input
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.2, 0]],  # little left
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-1, 0]],  # hard left
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.2, 0]],  # little right
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[1, 0]],  # hard right
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, -1]],  # brake
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, 1]],  # accelerate
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.4, 1]],  # accelerate and left
            self.g.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.4, 1]],  # accelerate and right
        ]

        return control_inputs

    def sample_control_inputs(self):
        """
        Sample zero mean gaussian control sequences
        @returns: results{list<control sequence>}: A zero mean gaussian distribution of control sequences
        """

        steering = np.random.normal(
            0,
            self.g.INITIAL_STEERING_VARIANCE,
            self.g.NUMBER_OF_INITIAL_TRAJECTORIES * self.g.NUMBER_OF_STEPS_PER_TRAJECTORY,
            )
        acceleration = np.random.normal(
            0,
            self.g.INITIAL_ACCELERATION_VARIANCE,
            self.g.NUMBER_OF_INITIAL_TRAJECTORIES * self.g.NUMBER_OF_STEPS_PER_TRAJECTORY,
            )

        control_input_sequences = np.column_stack((steering, acceleration))
        control_input_sequences = np.reshape(
            control_input_sequences,
            (self.g.NUMBER_OF_INITIAL_TRAJECTORIES, self.g.NUMBER_OF_STEPS_PER_TRAJECTORY, 2),
        )
        return control_input_sequences

    def sample_control_inputs_history_based(self, last_control_sequence):
        """
        Sample history based control sequences (simmilar to the last "perfect" strategy)
        @returns: results{list<control sequence>}: A history based small variance distribution of control sequences
        """

        # Chose sampling method by uncommenting
        return self.sample_control_inputs()
        # return self.static_control_inputs()

        # Not initialized
        if len(last_control_sequence) == 0:
            return self.sample_control_inputs()

        # Delete the first step of the last control sequence because it is already done
        # To keep the length of the control sequence add one to the end
        last_control_sequence = last_control_sequence[1:]
        last_control_sequence = np.append(last_control_sequence, [0, 0]).reshape(
            NUMBER_OF_STEPS_PER_TRAJECTORY, 2
        )

        last_steerings = last_control_sequence[:, 0]
        last_accelerations = last_control_sequence[:, 1]

        control_input_sequences = np.zeros(
            [NUMBER_OF_TRAJECTORIES, NUMBER_OF_STEPS_PER_TRAJECTORY, 2]
        )

        for i in range(NUMBER_OF_TRAJECTORIES):
            steering_noise = np.random.normal(
                0, STEP_STEERING_VARIANCE, NUMBER_OF_STEPS_PER_TRAJECTORY
            )
            acceleration_noise = np.random.normal(
                0, STEP_ACCELERATION_VARIANCE, NUMBER_OF_STEPS_PER_TRAJECTORY
            )

            next_steerings = last_steerings + steering_noise
            next_accelerations = last_accelerations + acceleration_noise

            # Optional: Filter for smoother control
            # next_steerings = signal.medfilt(next_steerings, 3)
            # next_accelerations = signal.medfilt(next_accelerations, 3)

            next_control_inputs = np.vstack((next_steerings, next_accelerations)).T
            control_input_sequences[i] = next_control_inputs

        return control_input_sequences

    def cost_function(self, trajectory):
        """
        calculate the cost of a trajectory

        :param trajectory: {list<state>} The trajectory of states that needs to be evaluated

        :returns: cost {float}: the scalar cost of the trajectory
        """

        self.g:neural_mpc_settings=reload_class_if_modified(self.g,every=100)

        distance_cost = 0
        angle_cost = 0
        terminal_speed_cost = 0
        terminal_position_cost = 0

        number_of_states = len(trajectory)
        index = 0

        angles = np.absolute(self.track.AngleNextCheckpointRelative)
        waypoint_index = self.track.get_nearest_waypoint_idx(x=self.car_state[0], y=self.car_state[1])
        angles = angles[
                 waypoint_index
                 + self.g.ANGLE_COST_INDEX_START: waypoint_index
                                                  + self.g.ANGLE_COST_INDEX_STOP
                 ]
        angles_squared = np.absolute(angles)  # np.square(angles)
        angle_sum = np.sum(angles_squared)

        for state in trajectory:
            # discount = (number_of_states - 0.1 * index) / number_of_states
            # Discount is no longer used

            simulated_position = geom.Point(state[0], state[1])
            distance_to_track = simulated_position.distance(self.trackline)

            distance_to_track = distance_to_track ** 2 if distance_to_track < 100 else 10000  # handle overflow

            # Don't leave track!
            if distance_to_track > self.g.TRACK_WIDTH:
                distance_cost += self.g.MAX_COST
            index += 1

            if distance_to_track < self.g.TRACK_WIDTH / 3.:
                distance_cost = distance_cost / 3.
            index += 1

        # Terminal Speed cost
        terminal_state = trajectory[-1]
        terminal_speed = terminal_state[3]
        terminal_speed_cost += abs(1 / terminal_speed)

        if terminal_state[3] < self.g.MIN_SPEED_MPS:  # Min speed  = 5
            terminal_speed_cost += 3 * abs(self.g.MIN_SPEED_MPS - terminal_speed)

        # Terminal Position cost
        terminal_position = geom.Point(terminal_state[0], terminal_state[1])
        terminal_distance_to_track = terminal_position.distance(self.trackline)
        terminal_position_cost += abs(terminal_distance_to_track)

        # Angle cost
        angle_cost = angle_sum * terminal_state[3]

        # Total cost
        weighted_distance_cost = self.g.DISTANCE_FROM_CENTERLINE_COST_WEIGHT * distance_cost
        weighted_speed_cost = self.g.TERMINAL_SPEED_COST_WEIGHT * terminal_speed_cost
        weighted_position_cost = self.g.TERMINAL_POSITION_COST_WEIGHT * terminal_position_cost
        weighted_angle_cost = self.g.ANGLE_COST_WEIGHT * angle_cost
        cost = (
                weighted_distance_cost
                + weighted_speed_cost
                + weighted_position_cost
                + weighted_angle_cost
        )
        self.counter+=1
        if self.stats is not None:
            self.stats.add(cost, weighted_distance_cost,weighted_speed_cost,weighted_position_cost,weighted_angle_cost)
        # if self.counter%10000==0:
        #     logger.info(f'cost {cost:.2n} = dist {weighted_distance_cost:.2n} + speed {weighted_speed_cost:.2n} + pos {weighted_position_cost:.1n} + angle {weighted_angle_cost:.2n}')
        if self.counter%self.stats.size==0:
            self.stats.compute_and_show()
        return cost

    def control_step(self):
        """
        Calculate an aprocimation to the optimal next control sequence

        :returns: next_control_sequence{list<control input>} The optimal control sequnce
        """

        self.update_trackline()

        # Do not overshoot the car's steering contraints
        self.last_control_input[0] = min(self.last_control_input[0], 1)
        self.last_control_input[1] = min(self.last_control_input[1], 0.5)

        # History based sampling due to the last optimal control sequence
        control_sequences = self.sample_control_inputs_history_based(
            self.best_control_sequenct
        )

        simulated_history, costs = self.simulate_trajectory_distribution(
            control_sequences
        )

        lowest_cost = 100000
        best_index = 0

        weights = np.zeros(len(control_sequences))

        for i in range(len(control_sequences)):
            cost = costs[i]
            # find best
            if cost < lowest_cost:
                best_index = i
                lowest_cost = cost

            if cost < self.g.MAX_COST:
                weight = math.exp((-1 / self.g.INVERSE_TEMP) * cost)
            else:
                weight = 0
            weights[i] = weight

        best_conrol_sequence = control_sequences[best_index]

        # Finding weighted avg input
        # If all trajectories are bad (weights = 0), go for the best one
        if weights.max() != 0:
            next_control_sequence = np.average(
                control_sequences, axis=0, weights=weights
            )
        else:
            next_control_sequence = best_conrol_sequence

        # Optional, just take best anyway
        next_control_sequence = best_conrol_sequence

        self.best_control_sequenct = next_control_sequence
        return next_control_sequence

    """
    draws the simulated history (position and speed) of the car into a fig for a trajectory distribution resp. the history of all trajectory distributions
    """

    def draw_simulated_history(self, waypoint_index=0, chosen_trajectory=[]):

        plt.clf()

        fig, position_ax = plt.subplots()

        plt.title("History based random control")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

        s_x = []
        s_y = []
        costs = []
        i = 0
        ind = 0
        indices = []
        for trajectory in self.simulated_history:
            cost = self.simulated_costs[i]
            if cost < self.g.MAX_COST:
                for state in trajectory:
                    if state[0] > 1:
                        s_x.append(state[0])
                        s_y.append(state[1])
                        costs.append(cost)
                        indices.append(cost)
                        ind += 1
            i += 1

        trajectory_costs = position_ax.scatter(s_x, s_y, c=indices)
        colorbar = fig.colorbar(trajectory_costs)
        colorbar.set_label("Trajectory costs")

        # Draw car position
        p_x = self.car_state[0]
        p_y = self.car_state[1]
        position_ax.scatter(p_x, p_y, c="#FF0000", label="Current car position")

        # Draw waypoints
        waypoint_index = self.track.get_nearest_waypoint_idx(x=self.car_state[0], y=self.car_state[1])

        waypoints = np.array(self.track.waypoints)
        w_x = waypoints[
              waypoint_index
              + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS: waypoint_index
                                                            + self.g.NUMBER_OF_NEXT_WAYPOINTS
                                                            + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
              0,
              ]
        w_y = waypoints[
              waypoint_index
              + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS: waypoint_index
                                                            + self.g.NUMBER_OF_NEXT_WAYPOINTS
                                                            + self.g.NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
              1,
              ]
        position_ax.scatter(w_x, w_y, c="#000000", label="Next waypoints")

        # Plot Chosen Trajectory
        t_x = []
        t_y = []
        for state in chosen_trajectory:
            t_x.append(state[0])
            t_y.append(state[1])

        plt.scatter(t_x, t_y, c="#D94496", label="Chosen control")
        plt.legend(fancybox=True, shadow=True, loc="best")

        plt.savefig("live_rollouts.png")
        return plt


    class mpc_stats():
        def __init__(self, outer, size=1000):
            """ Initialize the statistics
            :param outer: the containing instance
            :param size: how many stats to keep in circular RingBuffer
            """
            self.size=size
            self.costs=deque(maxlen=size)
            self.dist_costs=deque(maxlen=size)
            self.speed_costs=deque(maxlen=size)
            self.position_costs=deque(maxlen=size)
            self.angle_costs=deque(maxlen=size)
            self.fig:matplotlib.pyplot.figure=None
            self.outer=outer

        def add(self,cost, dist, speed, pos, ang):
            self.costs.append(cost)
            self.dist_costs.append(dist)
            self.speed_costs.append(speed)
            self.position_costs.append(pos)
            self.angle_costs.append(ang)

        def compute_and_show(self):
            self.outer.car.client.set_paused(True)
            # logger.info(f'plotting {len(self.costs)} samples')
            # if self.fig is None:
            #     self.fig, self.ax = plt.subplots()
            # else:
            #     self.fig.clf()
            # log=False
            # def pl(c,label=''): # https://stackoverflow.com/questions/28398200/matplotlib-plotting-transparent-histogram-with-non-transparent-edge
            #     # increment and get the "props" cycle (and extract the color)
            #     color = next(self.ax._get_lines.prop_cycler)["color"]
            #     # 1. draw: inner area with alpha
            #     self.ax.hist(np.array(c) ,bins=100,label=label,color=color, alpha=0.3)
            # pl(self.costs,'cost')
            # pl(self.dist_costs,'dist')
            # pl(self.speed_costs,'speed')
            # pl(self.position_costs,'position')
            # pl(self.angle_costs,'angle')
            # self.ax.legend()
            # self.ax.set_xlabel('cost')
            # self.ax.set_ylabel('frequency')
            # self.fig.show()
            def meas(c):
                a=np.array(c)
                mean=np.mean(a)
                std=np.std(a)
                # med=np.median(a)
                return f'{mean:.2f} ± {std:.2f}'
            all=[[self.costs,'cost'],[self.dist_costs,'dist'],[self.speed_costs,'speed'],[self.position_costs,'pos'],[self.angle_costs,'angle']]
            s='Costs '
            for a in all:
                s=s+a[1]+': '+meas(a[0])+' '
            logger.info(f'\n{s}')

            self.outer.car.client.set_paused(False)


