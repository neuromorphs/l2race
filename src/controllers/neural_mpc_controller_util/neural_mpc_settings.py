##########################################
########         Constants        ########
##########################################

# they are class to allow reloading class during runtime

class neural_mpc_settings():
    """
    Setting values for controller
    """
    T_CONTROL = 0.2  # TODO ??
    T_EULER_STEP = 0.01  # TODO ??

    MIN_SPEED_MPS = 5.0  # minimum speed of car that controller will generate control (model is not trained below this speed)

    # controller uses this controller where neural_mpc_controller is not trained
    LOW_SPEED_CONTROLLER_MODULE, LOW_SPEED_CONTROLLER_CLASS = 'src.controllers.pure_pursuit_controller_v2', 'pure_pursuit_controller_v2'

    ##########################################
    #######         Experiments        #######
    ##########################################

    DRAW_LIVE_ROLLOUTS = False  # TODO ?

    ##########################################
    #########       Car & Track     ##########
    ##########################################

    TRACK_WIDTH = 2  # TODO why not from track info?

    ##########################################
    ####   Neural MPC Car Controller     #####
    ##########################################

    # Cost function weights
    DISTANCE_FROM_CENTERLINE_COST_WEIGHT = 1
    TERMINAL_SPEED_COST_WEIGHT = 5000
    TERMINAL_POSITION_COST_WEIGHT = 3
    ANGLE_COST_WEIGHT = 2 # weights

    # Path Prediction
    CONTROLLER_PREDICTIOR = "nn"
    CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-high-speed"  # For large race track: M_PER_PIXEL = 0.2, speed up to 25m/s
    # CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-invariant-10"  # For small race track: M_PER_PIXEL = 0.1, speed up to 15m/s

    NUMBER_OF_STEPS_PER_TRAJECTORY = 10  # MPC horizon in steps TODO what are these?
    INVERSE_TEMP = 5 # TODO ??

    # Initializing parameters
    NUMBER_OF_INITIAL_TRAJECTORIES = 150  # number of rollouts per control step
    INITIAL_STEERING_VARIANCE = 0.25  # TODO what are units?
    INITIAL_ACCELERATION_VARIANCE = 0.4  # TODO units?  What is typical range?

    # Parameters for rollout
    # Not used: only for history based sampling
    # NUMBER_OF_TRAJECTORIES = 200
    # STEP_STEERING_VARIANCE = 0.1
    # STEP_ACCELERATION_VARIANCE = 0.1

    # Relation to track
    NUMBER_OF_NEXT_WAYPOINTS = 20  # TODO how related to horizon?
    NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 5 # TODO what is effect?
    ANGLE_COST_INDEX_START = 5  # TODO ??
    ANGLE_COST_INDEX_STOP = 15  # TODO ??

    # Relations to car
    MAX_COST = 1000  # Cost that is set to infinity (weights = 0)

    ##########################################
    #########       NN Training     ##########
    ##########################################

    # Not used in l2race. For model training please use CommonRoadMPC
    # https://github.com/neuromorphs/CommonRoadMPC

    # Artificial data generation
    # The training data is saved/retreived in nn_prediction/training/data/[filename]
    # DATA_GENERATION_FILE = "training_data_sequential.csv"

    # Training parameters
    MODEL_NAME = "Dense-128-128-128-128-sequential"
    # TRAINING_DATA_FILE = "training_data_sequential.csv"
    # NUMBER_OF_EPOCHS = 150
    # BATCH_SIZE = 128
    # PREDICT_DELTA = True
    # NORMALITE_DATA = True
    # CUT_INVARIANTS = True
