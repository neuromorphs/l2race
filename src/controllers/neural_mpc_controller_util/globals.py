##########################################
##########         Global        #########
##########################################

T_CONTROL = 0.2
T_EULER_STEP = 0.01


##########################################
#######         Experiments        #######
##########################################

SIMULATION_LENGTH = 500
DRAW_LIVE_HISTORY = True
DRAW_LIVE_ROLLOUTS = False
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"



##########################################
#########       Car & Track     ##########
##########################################

INITIAL_SPEED = 8
CONTINUE_FROM_LAST_STATE = False

TRACK_NAME = "track_2"
M_TO_PIXEL = 0.1
TRACK_WIDTH = 2


##########################################
#########     Car Controller     #########
##########################################

# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 150
INITIAL_STEERING_VARIANCE = 0.4
INITIAL_ACCELERATION_VARIANCE = 0.4


# Parameters for rollout
NUMBER_OF_TRAJECTORIES = 200
STRATEGY_COVARIANCE = [[0.3, 0], [0, 0.3]]
NUMBER_OF_STEPS_PER_TRAJECTORY = 10
INVERSE_TEMP = 5

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 117
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 2
DIST_TOLLERANCE = 4

# Relations to car
MAX_SPEED = 25
MAX_COST = 1000


##########################################
#########       NN Training     ##########
##########################################

# Training parameters
MODEL_NAME = "Dense-128-128-128-128-uniform-40"
TRAINING_DATA_FILE = "training_data_1000x1000x10.csv"
NUMBER_OF_EPOCHS = 40
BATCH_SIZE = 128
PREDICT_DELTA = True
NORMALITE_DATA = True
