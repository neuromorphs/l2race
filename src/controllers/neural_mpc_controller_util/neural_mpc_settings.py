##########################################
########         Constants        ########
##########################################

T_CONTROL = 0.2
T_EULER_STEP = 0.01


##########################################
#######         Experiments        #######
##########################################

DRAW_LIVE_ROLLOUTS = False


##########################################
#########       Car & Track     ##########
##########################################

TRACK_WIDTH = 2


##########################################
####   Neural MPC Car Controller     #####
##########################################

# Path Prediction
CONTROLLER_PREDICTIOR = "nn"
CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-high-speed"  # For large race track: M_PER_PIXEL = 0.2, speed up to 25m/s
# CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-invariant-10"  # For small race track: M_PER_PIXEL = 0.1, speed up to 15m/s

NUMBER_OF_STEPS_PER_TRAJECTORY = 10 #MPC horizon
INVERSE_TEMP = 5

# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 150
INITIAL_STEERING_VARIANCE = 0.5
INITIAL_ACCELERATION_VARIANCE = 0.4


# Parameters for rollout
# Not used: only for history based sampling
# NUMBER_OF_TRAJECTORIES = 200
# STEP_STEERING_VARIANCE = 0.1
# STEP_ACCELERATION_VARIANCE = 0.1

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 20
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 3
ANGLE_COST_INDEX_START = 5
ANGLE_COST_INDEX_STOP = 15

# Relations to car
MAX_COST = 1000 #Cost that is set to infinity (weights = 0)


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
