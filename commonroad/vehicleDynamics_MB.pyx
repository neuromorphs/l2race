import logging

from .steeringConstraints import steeringConstraints
from .accelerationConstraints import accelerationConstraints
from .vehicleDynamics_KS import vehicleDynamics_KS
from . import vehicleParameters
from . import tireModel
import math
from typing import *
# from .vehicleParameters import VehicleParameters, vehicle_params_type
from src.my_logger import my_logger
from cpython cimport array
import array
logger = my_logger(__name__)
logger.setLevel(logging.DEBUG)

cdef float KS_SWITCH_SPEED=0.1

import cython
if cython.compiled:
    logger.info("check_cython: {} is compiled Cython.".format(__file__))
else:
    logger.warning("check_cython: {} is still just a slowly interpreted script.".format(__file__))

cpdef double[:] vehicleDynamics_MB(double[:] x,double[:] uInit,object p):
    # vehicleDynamics_MB - multi-body vehicle dynamics based on the DOT 
    # (department of transportation) vehicle dynamics
    #
    # Syntax:  
    #    f = vehicleDynamics_MB(t,x,u,p)
    #
    # Inputs:
    #    x - vehicle state vector
    #    u - vehicle input vector
    #    p - vehicle parameter vector
    #
    # Outputs:
    #    f - right-hand side of differential equations
    #
    # Example: 
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      05-January-2017
    # Last update:17-December-2017
    # Last revision:---

    #------------- BEGIN CODE --------------
    # set gravity constant
    cdef float g = 9.81  #[m/s^2]

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle
    #x6 = yaw rate

    #x7 = roll angle
    #x8 = roll rate
    #x9 = pitch angle
    #x10 = pitch rate
    #x11 = velocity in y-direction
    #x12 = z-position
    #x13 = velocity in z-direction

    #x14 = roll angle front
    #x15 = roll rate front
    #x16 = velocity in y-direction front
    #x17 = z-position front
    #x18 = velocity in z-direction front

    #x19 = roll angle rear
    #x20 = roll rate rear
    #x21 = velocity in y-direction rear
    #x22 = z-position rear
    #x23 = velocity in z-direction rear

    #x24 = left front wheel angular speed
    #x25 = right front wheel angular speed
    #x26 = left rear wheel angular speed
    #x27 = right rear wheel angular speed

    #x28 = delta_y_f
    #x29 = delta_y_r

    #u1 = steering angle velocity of front wheels
    #u2 = acceleration

    #consider steering constraints
    cdef array.array u=array.array('d',[steeringConstraints(x[2],uInit[0],p.steering),
        accelerationConstraints(x[3],uInit[1],p.longitudinal)])

    #compute slip angle at cg
    #switch to kinematic model for small velocities
    cdef float beta
    if x[3]<0 or abs(x[3]) < KS_SWITCH_SPEED:
        beta = 0
    else:
        beta = math.atan(x[10]/x[3]) 
    cdef float speed = math.sqrt(x[3]**2 + x[10]**2)

    #vertical tire forces
    cdef float F_z_LF = (x[16] + p.R_w*(math.cos(x[13]) - 1) - 0.5*p.T_f*math.sin(x[13]))*p.K_zt
    cdef float F_z_RF = (x[16] + p.R_w*(math.cos(x[13]) - 1) + 0.5*p.T_f*math.sin(x[13]))*p.K_zt
    cdef float F_z_LR = (x[21] + p.R_w*(math.cos(x[18]) - 1) - 0.5*p.T_r*math.sin(x[18]))*p.K_zt
    cdef float F_z_RR = (x[21] + p.R_w*(math.cos(x[18]) - 1) + 0.5*p.T_r*math.sin(x[18]))*p.K_zt

    #obtain individual tire speeds
    cdef float u_w_lf = (x[3] + 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2])
    cdef float u_w_rf = (x[3] - 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2])
    cdef float u_w_lr = x[3] + 0.5*p.T_r*x[5]
    cdef float u_w_rr = x[3] - 0.5*p.T_r*x[5]

    cdef float s_lf
    cdef float     s_rf
    cdef float     s_lr
    cdef float     s_rr

    #compute longitudinal slip
    #switch to kinematic model for small velocities
    if x[3]<0 or abs(x[3]) < 2.0:
        s_lf = 0
        s_rf = 0
        s_lr = 0
        s_rr = 0
    else:
        s_lf = 1 - p.R_w*x[23]/u_w_lf 
        s_rf = 1 - p.R_w*x[24]/u_w_rf 
        s_lr = 1 - p.R_w*x[25]/u_w_lr 
        s_rr = 1 - p.R_w*x[26]/u_w_rr 

    #lateral slip angles
    #switch to kinematic model for small velocities
    cdef float     alpha_LF
    cdef float     alpha_RF
    cdef float    alpha_LR
    cdef float     alpha_RR
    if abs(x[3]) < KS_SWITCH_SPEED:
        alpha_LF = 0
        alpha_RF = 0
        alpha_LR = 0
        alpha_RR = 0
    else:
        alpha_LF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] + 0.5*p.T_f*x[5])) - x[2]
        alpha_RF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] - 0.5*p.T_f*x[5])) - x[2] 
        alpha_LR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] + 0.5*p.T_r*x[5])) 
        alpha_RR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] - 0.5*p.T_r*x[5])) 

    #auxiliary suspension movement
    cdef float     z_SLF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] + 0.5*(x[6] - x[13])*p.T_f
    cdef float     z_SRF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] - 0.5*(x[6] - x[13])*p.T_f
    cdef float     z_SLR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] + 0.5*(x[6] - x[18])*p.T_r
    cdef float     z_SRR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] - 0.5*(x[6] - x[18])*p.T_r

    cdef float     dz_SLF = x[17] - x[12] + p.a*x[9] + 0.5*(x[7] - x[14])*p.T_f
    cdef float     dz_SRF = x[17] - x[12] + p.a*x[9] - 0.5*(x[7] - x[14])*p.T_f
    cdef float     dz_SLR = x[22] - x[12] - p.b*x[9] + 0.5*(x[7] - x[19])*p.T_r
    cdef float     dz_SRR = x[22] - x[12] - p.b*x[9] - 0.5*(x[7] - x[19])*p.T_r

    #camber angles
    cdef float     gamma_LF = x[6] + p.D_f*z_SLF + p.E_f*(z_SLF)**2
    cdef float     gamma_RF = x[6] - p.D_f*z_SRF - p.E_f*(z_SRF)**2
    cdef float     gamma_LR = x[6] + p.D_r*z_SLR + p.E_r*(z_SLR)**2
    cdef float     gamma_RR = x[6] - p.D_r*z_SRR - p.E_r*(z_SRR)**2

    #compute longitudinal tire forces using the magic formula for pure slip
    cdef float     F0_x_LF = tireModel.mFormulaLongitudinal(s_lf, gamma_LF, F_z_LF, p.tire)
    cdef float     F0_x_RF = tireModel.mFormulaLongitudinal(s_rf, gamma_RF, F_z_RF, p.tire)
    cdef float     F0_x_LR = tireModel.mFormulaLongitudinal(s_lr, gamma_LR, F_z_LR, p.tire)
    cdef float     F0_x_RR = tireModel.mFormulaLongitudinal(s_rr, gamma_RR, F_z_RR, p.tire)

    #compute lateral tire forces using the magic formula for pure slip
    cdef double[:]     res = tireModel.mFormulaLateral(alpha_LF, gamma_LF, F_z_LF, p.tire)
    cdef float     F0_y_LF = res[0]
    cdef float     mu_y_LF = res[1]
    res = tireModel.mFormulaLateral(alpha_RF, gamma_RF, F_z_RF, p.tire) 
    cdef float     F0_y_RF = res[0]
    cdef float     mu_y_RF = res[1]
    res = tireModel.mFormulaLateral(alpha_LR, gamma_LR, F_z_LR, p.tire) 
    cdef float     F0_y_LR = res[0]
    cdef float     mu_y_LR = res[1]
    res = tireModel.mFormulaLateral(alpha_RR, gamma_RR, F_z_RR, p.tire) 
    cdef float     F0_y_RR = res[0]
    cdef float     mu_y_RR = res[1]

    #compute longitudinal tire forces using the magic formula for combined slip
    cdef float     F_x_LF = tireModel.mFormulaLongitudinalComb(s_lf, alpha_LF, F0_x_LF, p.tire)
    cdef float     F_x_RF = tireModel.mFormulaLongitudinalComb(s_rf, alpha_RF, F0_x_RF, p.tire)
    cdef float     F_x_LR = tireModel.mFormulaLongitudinalComb(s_lr, alpha_LR, F0_x_LR, p.tire)
    cdef float     F_x_RR = tireModel.mFormulaLongitudinalComb(s_rr, alpha_RR, F0_x_RR, p.tire)

    #compute lateral tire forces using the magic formula for combined slip
    cdef float     F_y_LF = tireModel.mFormulaLateralComb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, p.tire)
    cdef float     F_y_RF = tireModel.mFormulaLateralComb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire)
    cdef float     F_y_LR = tireModel.mFormulaLateralComb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, p.tire)
    cdef float     F_y_RR = tireModel.mFormulaLateralComb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire)

    #auxiliary movements for compliant joint equations
    cdef float     delta_z_f = p.h_s - p.R_w + x[16] - x[11]
    cdef float     delta_z_r = p.h_s - p.R_w + x[21] - x[11]

    cdef float     delta_phi_f = x[6] - x[13]
    cdef float     delta_phi_r = x[6] - x[18]

    cdef float     dot_delta_phi_f = x[7] - x[14]
    cdef float     dot_delta_phi_r = x[7] - x[19]

    cdef float     dot_delta_z_f = x[17] - x[12]
    cdef float     dot_delta_z_r = x[22] - x[12]

    cdef float     dot_delta_y_f = x[10] + p.a*x[5] - x[15]
    cdef float     dot_delta_y_r = x[10] - p.b*x[5] - x[20]

    cdef float     delta_f = delta_z_f*math.sin(x[6]) - x[27]*math.cos(x[6]) - (p.h_raf - p.R_w)*math.sin(delta_phi_f)
    cdef float     delta_r = delta_z_r*math.sin(x[6]) - x[28]*math.cos(x[6]) - (p.h_rar - p.R_w)*math.sin(delta_phi_r)

    cdef float     dot_delta_f = (delta_z_f*math.cos(x[6]) + x[27]*math.sin(x[6]))*x[7] + dot_delta_z_f*math.sin(x[6]) - dot_delta_y_f*math.cos(x[6]) - (p.h_raf - p.R_w)*math.cos(delta_phi_f)*dot_delta_phi_f
    cdef float     dot_delta_r = (delta_z_r*math.cos(x[6]) + x[28]*math.sin(x[6]))*x[7] + dot_delta_z_r*math.sin(x[6]) - dot_delta_y_r*math.cos(x[6]) - (p.h_rar - p.R_w)*math.cos(delta_phi_r)*dot_delta_phi_r
              
    #compliant joint forces
    cdef float     F_RAF = delta_f*p.K_ras + dot_delta_f*p.K_rad
    cdef float     F_RAR = delta_r*p.K_ras + dot_delta_r*p.K_rad

    #auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    cdef float     F_SLF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SLF*p.K_sf - dz_SLF*p.K_sdf + (x[6] - x[13])*p.K_tsf/p.T_f

    cdef float     F_SRF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SRF*p.K_sf - dz_SRF*p.K_sdf - (x[6] - x[13])*p.K_tsf/p.T_f

    cdef float     F_SLR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SLR*p.K_sr - dz_SLR*p.K_sdr + (x[6] - x[18])*p.K_tsr/p.T_r

    cdef float     F_SRR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SRR*p.K_sr - dz_SRR*p.K_sdr - (x[6] - x[18])*p.K_tsr/p.T_r


    #auxiliary variables sprung mass
    cdef float     sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF)*math.cos(x[2]) - (F_y_LF + F_y_RF)*math.sin(x[2])

    cdef float     sumN = (F_y_LF + F_y_RF)*p.a*math.cos(x[2]) + (F_x_LF + F_x_RF)*p.a*math.sin(x[2]) \
           + (F_y_RF - F_y_LF)*0.5*p.T_f*math.sin(x[2]) + (F_x_LF - F_x_RF)*0.5*p.T_f*math.cos(x[2]) \
           + (F_x_LR - F_x_RR)*0.5*p.T_r - (F_y_LR + F_y_RR)*p.b 
       
    cdef float     sumY_s = (F_RAF + F_RAR)*math.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR)*math.sin(x[6])

    cdef float     sumL = 0.5*F_SLF*p.T_f + 0.5*F_SLR*p.T_r - 0.5*F_SRF*p.T_f - 0.5*F_SRR*p.T_r \
           - F_RAF/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[16] - (p.h_raf - p.R_w)*math.cos(x[13])) \
           - F_RAR/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[21] - (p.h_rar - p.R_w)*math.cos(x[18])) 
       
    cdef float     sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR)*math.cos(x[6]) - (F_RAF + F_RAR)*math.sin(x[6])

    cdef float     sumM_s = p.a*(F_SLF + F_SRF) - p.b*(F_SLR + F_SRR) + ((F_x_LF + F_x_RF)*math.cos(x[2]) \
           - (F_y_LF + F_y_RF)*math.sin(x[2]) + F_x_LR + F_x_RR)*(p.h_s - x[11]) 

    #auxiliary variables unsprung mass
    cdef float     sumL_uf = 0.5*F_SRF*p.T_f - 0.5*F_SLF*p.T_f - F_RAF*(p.h_raf - p.R_w) \
              + F_z_LF*(p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) - p.K_lt*F_y_LF) \
              - F_z_RF*(-p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) + p.K_lt*F_y_RF) \
              - ((F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]))*(p.R_w - x[16]) 
          
    cdef float     sumL_ur = 0.5*F_SRR*p.T_r - 0.5*F_SLR*p.T_r - F_RAR*(p.h_rar - p.R_w) \
              + F_z_LR*(p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) - p.K_lt*F_y_LR) \
              - F_z_RR*(-p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) + p.K_lt*F_y_RR) \
              - (F_y_LR + F_y_RR)*(p.R_w - x[21])    
          
    cdef float     sumZ_uf = F_z_LF + F_z_RF + F_RAF*math.sin(x[6]) - (F_SLF + F_SRF)*math.cos(x[6])

    cdef float     sumZ_ur = F_z_LR + F_z_RR + F_RAR*math.sin(x[6]) - (F_SLR + F_SRR)*math.cos(x[6])

    cdef float     sumY_uf = (F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]) \
              - F_RAF*math.cos(x[6]) - (F_SLF + F_SRF)*math.sin(x[6]) 
          
    cdef float     sumY_ur = (F_y_LR + F_y_RR) \
              - F_RAR*math.cos(x[6]) - (F_SLR + F_SRR)*math.sin(x[6])  
          
          
    #dynamics common with single-track model
    cdef array.array f=array.array('d') # init 'right hand side'
    #switch to kinematic model for small velocities
    if abs(x[3]) < KS_SWITCH_SPEED:
        #wheelbase
        lwb = p.a + p.b
        
        #system dynamics
        x_ks= [x[0],  x[1],  x[2],  x[3],  x[4]]
        f_ks= vehicleDynamics_KS(x_ks,u,p)
        f.extend(f_ks)
        f.append(u[1]*lwb*math.tan(x[2]) + x[3]/(lwb*math.cos(x[2])**2)*u[0])
    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle
    #x6 = yaw rate

    #x7 = roll angle
    #x8 = roll rate
    #x9 = pitch angle
    #x10 = pitch rate
    #x11 = velocity in y-direction
    #x12 = z-position
    #x13 = velocity in z-direction



    else:
        f.append(math.cos(beta + x[4])*speed)
        f.append(math.sin(beta + x[4])*speed)
        f.append(u[0])
        f.append(1/p.m*sumX + x[5]*x[10])
        f.append(x[5])
        f.append(1/(p.I_z - (p.I_xz_s)**2/p.I_Phi_s)*(sumN + p.I_xz_s/p.I_Phi_s*sumL))


    # remaining sprung mass dynamics
    f.append(x[7])
    f.append(1/(p.I_Phi_s - (p.I_xz_s)**2/p.I_z)*(p.I_xz_s/p.I_z*sumN + sumL))
    f.append(x[9])
    f.append(1/p.I_y_s*sumM_s)
    f.append(1/p.m_s*sumY_s - x[5]*x[3])
    f.append(x[12])
    f.append(g - 1/p.m_s*sumZ_s)

    #unsprung mass dynamics (front)
    f.append(x[14])
    f.append(1/p.I_uf*sumL_uf)
    f.append(1/p.m_uf*sumY_uf - x[5]*x[3])
    f.append(x[17])
    f.append(g - 1/p.m_uf*sumZ_uf)

    #unsprung mass dynamics (rear)
    f.append(x[19])
    f.append(1/p.I_ur*sumL_ur)
    f.append(1/p.m_ur*sumY_ur - x[5]*x[3])
    f.append(x[22])
    f.append(g - 1/p.m_ur*sumZ_ur)

    #convert acceleration input to brake and engine torque
    cdef float     T_B, T_E
    if u[1]>0:
        T_B = 0 
        T_E = p.m*p.R_w*u[1] 
    else:
        T_B = p.m*p.R_w*u[1] 
        T_E = 0 

    #wheel dynamics (p.T  new parameter for torque splitting)
    f.append(1/p.I_y_w*(-p.R_w*F_x_LF + 0.5*p.T_sb*T_B + 0.5*p.T_se*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RF + 0.5*p.T_sb*T_B + 0.5*p.T_se*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_LR + 0.5*(1-p.T_sb)*T_B + 0.5*(1-p.T_se)*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RR + 0.5*(1-p.T_sb)*T_B + 0.5*(1-p.T_se)*T_E))

    #negative wheel spin forbidden
    for iState in range(23, 26):
        if x[iState]<0:
           x[iState] = 0 
           f[iState] = 0  

    #compliant joint equations
    f.append(dot_delta_y_f)
    f.append(dot_delta_y_r)

    return f

    #------------- END OF CODE --------------
