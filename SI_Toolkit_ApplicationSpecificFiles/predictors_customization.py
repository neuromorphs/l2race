import numpy as np
import yaml, os # TODO revamp this script, maybe remove altogether

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config_training.yml'), 'r'), Loader=yaml.FullLoader)

STATE_VARIABLES = config['training_default']['state_inputs']
STATE_INDICES = {k: v for v, k in enumerate(STATE_VARIABLES)}
CONTROL_INPUTS = config['training_default']['control_inputs']

def augment_predictor_output(output_array, net_info):

    pass

    return output_array

def next_state_predictor_ODE(dt, intermediate_steps):

    pass

    return output_array
