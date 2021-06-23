import numpy as np


def augment_data(x,y):

    x,y = shift_x0_invariant(x,y)
    x,y = shift_x1_invariant(x,y)

    return x,y



def shift_x0_invariant(x, y):

    x_augumented = x.copy()
    y_augumented = y.copy()
    
    for i in range(1,11):
        x_shifted,y_shifted  = shift_variable(x,y,0,i)
        x_augumented = np.concatenate((x_augumented,x_shifted), axis = 0)
        y_augumented = np.concatenate((y_augumented,y_shifted), axis = 0)
  
    return x_augumented,y_augumented

def shift_x1_invariant(x, y):

    x_augumented = x.copy()
    y_augumented = y.copy()
    
    for i in range(1,5):
        x_shifted,y_shifted  = shift_variable(x,y,1,i)
        x_augumented = np.concatenate((x_augumented,x_shifted), axis = 0)
        y_augumented = np.concatenate((y_augumented,y_shifted), axis = 0)
  
    return x_augumented,y_augumented






def shift_variable(x, y, index, distance):

    # We dont want to change the original value
    x = x.copy()
    y = y.copy()

    offset = np.array( len(x) * [distance]) 

    x[:,index] += offset
    y[:,index] += offset

    return x,y

