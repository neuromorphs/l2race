from numba.experimental import jitclass
from numba import float64  #

spec = [
    ('p_cx1', float64),
    ('p_dx1', float64),
    ('p_dx3', float64),
    ('p_ex1', float64),
    ('p_kx1', float64),
    ('p_hx1', float64),
    ('p_vx1', float64),
    ('r_bx1', float64),
    ('r_bx2', float64),
    ('r_cx1', float64),
    ('r_ex1', float64),
    ('r_hx1', float64),
    ('p_cy1', float64),
    ('p_dy1', float64),
    ('p_dy3', float64),
    ('p_ey1', float64),
    ('p_ky1', float64),
    ('p_hy1', float64),
    ('p_hy3', float64),
    ('p_vy1', float64),
    ('p_vy3', float64),
    ('r_by1', float64),
    ('r_by2', float64),
    ('r_by3', float64),
    ('r_cy1', float64),
    ('r_ey1', float64),
    ('r_hy1', float64),
    ('r_hy1', float64),
    ('r_vy1', float64),
    ('r_vy3', float64),
    ('r_vy4', float64),
    ('r_vy5', float64),
    ('r_vy6', float64),
]
# @jitclass(spec)
class TireParameters():
    def __init__(self):
       #tire parameters from ADAMS handbook
        #longitudinal coefficients
        self.p_cx1 = 0.  #Shape factor Cfx for longitudinal force
        self.p_dx1 = 0.  #Longitudinal friction Mux at Fznom
        self.p_dx3 = 0.  #Variation of friction Mux with camber
        self.p_ex1 = 0.  #Longitudinal curvature Efx at Fznom
        self.p_kx1 = 0.  #Longitudinal slip stiffness Kfx/Fz at Fznom
        self.p_hx1 = 0.  #Horizontal shift Shx at Fznom
        self.p_vx1 = 0.  #Vertical shift Svx/Fz at Fznom
        self.r_bx1 = 0.  #Slope factor for combined slip Fx reduction
        self.r_bx2 = 0.  #Variation of slope Fx reduction with kappa
        self.r_cx1 = 0.  #Shape factor for combined slip Fx reduction
        self.r_ex1 = 0.  #Curvature factor of combined Fx
        self.r_hx1 = 0.  #Shift factor for combined slip Fx reduction

        #lateral coefficients
        self.p_cy1 = 0.  #Shape factor Cfy for lateral forces
        self.p_dy1 = 0.  #Lateral friction Muy
        self.p_dy3 = 0.  #Variation of friction Muy with squared camber
        self.p_ey1 = 0.  #Lateral curvature Efy at Fznom
        self.p_ky1 = 0.  #Maximum value of stiffness Kfy/Fznom
        self.p_hy1 = 0.  #Horizontal shift Shy at Fznom
        self.p_hy3 = 0.  #Variation of shift Shy with camber
        self.p_vy1 = 0.  #Vertical shift in Svy/Fz at Fznom
        self.p_vy3 = 0.  #Variation of shift Svy/Fz with camber
        self.r_by1 = 0.  #Slope factor for combined Fy reduction
        self.r_by2 = 0.  #Variation of slope Fy reduction with alpha
        self.r_by3 = 0.  #Shift term for alpha in slope Fy reduction
        self.r_cy1 = 0.  #Shape factor for combined Fy reduction
        self.r_ey1 = 0.  #Curvature factor of combined Fy
        self.r_hy1 = 0.  #Shift factor for combined Fy reduction
        self.r_vy1 = 0.  #Kappa induced side force Svyk/Muy*Fz at Fznom
        self.r_vy3 = 0.  #Variation of Svyk/Muy*Fz with camber
        self.r_vy4 = 0.  #Variation of Svyk/Muy*Fz with alpha
        self.r_vy5 = 0.  #Variation of Svyk/Muy*Fz with kappa
        self.r_vy6 = 0.  #Variation of Svyk/Muy*Fz with atan(kappa)
