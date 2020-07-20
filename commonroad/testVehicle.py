from parameters_vehicle1 import parameters_vehicle1
from init_KS import init_KS
from init_ST import init_ST
from init_MB import init_MB
from vehicleDynamics_KS import vehicleDynamics_KS
from vehicleDynamics_ST import vehicleDynamics_ST
from vehicleDynamics_MB import vehicleDynamics_MB
from scipy.integrate import odeint, solve_ivp
import numpy
import matplotlib.pyplot as plt


def func_KS(x, t, u, p):
    f = vehicleDynamics_KS(x, u, p)
    return f
    

def func_ST(x, t, u, p):
    f = vehicleDynamics_ST(x, u, p)
    return f


def func_MB(x, t, u, p):
    f = vehicleDynamics_MB(x, u, p)
    return f


#load parameters
p = parameters_vehicle1()
g = 9.81  #[m/s^2]

#set options --------------------------------------------------------------
tStart = 0  #start time
tFinal = 1  #start time

delta0 = 0
vel0 = 15 
Psi0 = 0 
dotPsi0 = 0 
beta0 = 0 
sy0 = 0 
initialState = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]  #initial state for simulation
x0_KS = init_KS(initialState)  #initial state for kinematic single-track model
x0_ST = init_ST(initialState)  #initial state for single-track model
x0_MB = init_MB(initialState, p)  #initial state for multi-body model
#--------------------------------------------------------------------------
#

t = numpy.arange(0,tFinal, 0.01)
u = [0, 0]
x = odeint(func_KS, x0_KS, t, args=(u, p))

print(x)

t = numpy.arange(0,tFinal, 0.01)
u = [0, 5]
x = odeint(func_KS, x0_KS, t, args=(u, p))

print(x)

#set input: rolling car (velocity should stay constant)
t = numpy.arange(0,tFinal, 0.01)
u = [0, 0] 
print(x0_MB)
#simulate car
x = odeint(func_MB, x0_MB, t, args=(u, p))
#plot velocity
plt.plot(t, [tmp[3] for tmp in x])
plt.show()

#set input: braking car (wheel spin and velocity should decrease  similar wheel spin)
v_delta = 0.15 
acc = -0.7*g 
u = [v_delta,  acc] 
#simulate car
x_brake = odeint(func_MB, x0_MB, t, args=(u, p))
#position
plt.plot([tmp[0] for tmp in x_brake], [tmp[1] for tmp in x_brake])
plt.show()
# velocity
plt.plot(t, [tmp[3] for tmp in x_brake])
plt.show()
# wheel spin
plt.plot(t, [tmp[23] for tmp in x_brake])
plt.plot(t, [tmp[24] for tmp in x_brake])
plt.plot(t, [tmp[25] for tmp in x_brake])
plt.plot(t, [tmp[26] for tmp in x_brake])
plt.show()
# pitch
plt.plot(t, [tmp[8] for tmp in x_brake])
plt.show()

#set input: accelerating car (wheel spin and velocity should increase  more wheel spin at rear)
v_delta = 0.15 
acc = 0.63*g 
u = [v_delta,  acc] 
#simulate car
x_acc = odeint(func_MB, x0_MB, t, args=(u, p))
#position
plt.plot([tmp[0] for tmp in x_acc], [tmp[1] for tmp in x_acc])
plt.show()
# velocity
plt.plot(t, [tmp[3] for tmp in x_acc])
plt.show()
# wheel spin
plt.plot(t, [tmp[23] for tmp in x_acc])
plt.plot(t, [tmp[24] for tmp in x_acc])
plt.plot(t, [tmp[25] for tmp in x_acc])
plt.plot(t, [tmp[26] for tmp in x_acc])
plt.show()
# pitch
plt.plot(t, [tmp[8] for tmp in x_acc])
plt.show()
#orientation
plt.plot(t, [tmp[4] for tmp in x_acc])
plt.show()


#steering to left
v_delta = 0.15 
u = [v_delta,  0] 

#simulate full car
x_left = odeint(func_MB, x0_MB, t, args=(u, p))

#simulate single-track model
x_left_st = odeint(func_ST, x0_ST, t, args=(u, p))

#simulate kinematic single-track model
x_left_ks = odeint(func_KS, x0_KS, t, args=(u, p))

#position
plt.plot([tmp[0] for tmp in x_left], [tmp[1] for tmp in x_left])
plt.plot([tmp[0] for tmp in x_left_st], [tmp[1] for tmp in x_left_st])
plt.plot([tmp[0] for tmp in x_left_ks], [tmp[1] for tmp in x_left_ks])
plt.show()
#orientation
plt.plot(t, [tmp[4] for tmp in x_left])
plt.plot(t, [tmp[4] for tmp in x_left_st])
plt.plot(t, [tmp[4] for tmp in x_left_ks])
plt.show()
#steering
plt.plot(t, [tmp[2] for tmp in x_left])
plt.plot(t, [tmp[2] for tmp in x_left_st])
plt.plot(t, [tmp[2] for tmp in x_left_ks])
plt.show()
#yaw rate
plt.plot(t, [tmp[5] for tmp in x_left])
plt.plot(t, [tmp[5] for tmp in x_left_st])
plt.show()
#slip angle
plt.plot(t, [tmp[10]/tmp[3] for tmp in x_left])
plt.plot(t, [tmp[6] for tmp in x_left_st])
plt.show()

# figure # wheel spin
# hold on
# plot(t_acc,x_acc(:,24)) 
# plot(t_acc,x_acc(:,25)) 
# plot(t_acc,x_acc(:,26)) 
# plot(t_acc,x_acc(:,27)) 


#compare position for braking/normal
#position
plt.plot([tmp[0] for tmp in x_left], [tmp[1] for tmp in x_left])
plt.plot([tmp[0] for tmp in x_brake], [tmp[1] for tmp in x_brake])
plt.plot([tmp[0] for tmp in x_acc], [tmp[1] for tmp in x_acc])
plt.show()
#compare slip angles
plt.plot(t, [tmp[10]/tmp[3] for tmp in x_left])
plt.plot(t, [tmp[10]/tmp[3] for tmp in x_brake])
plt.plot(t, [tmp[10]/tmp[3] for tmp in x_acc])
plt.show()
#orientation
plt.plot(t, [tmp[4] for tmp in x_left])
plt.plot(t, [tmp[4] for tmp in x_brake])
plt.plot(t, [tmp[4] for tmp in x_acc])
plt.show()
#pitch
plt.plot(t, [tmp[8] for tmp in x_left])
plt.plot(t, [tmp[8] for tmp in x_brake])
plt.plot(t, [tmp[8] for tmp in x_acc])
plt.show()

#------------- END OF CODE --------------
