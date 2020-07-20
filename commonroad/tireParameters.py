class TireParameters():
    def __init__(self):
       #tire parameters from ADAMS handbook
        #longitudinal coefficients
        self.p_cx1 = []  #Shape factor Cfx for longitudinal force
        self.p_dx1 = []  #Longitudinal friction Mux at Fznom
        self.p_dx3 = []  #Variation of friction Mux with camber
        self.p_ex1 = []  #Longitudinal curvature Efx at Fznom
        self.p_kx1 = []  #Longitudinal slip stiffness Kfx/Fz at Fznom
        self.p_hx1 = []  #Horizontal shift Shx at Fznom
        self.p_vx1 = []  #Vertical shift Svx/Fz at Fznom
        self.r_bx1 = []  #Slope factor for combined slip Fx reduction
        self.r_bx2 = []  #Variation of slope Fx reduction with kappa
        self.r_cx1 = []  #Shape factor for combined slip Fx reduction
        self.r_ex1 = []  #Curvature factor of combined Fx
        self.r_hx1 = []  #Shift factor for combined slip Fx reduction

        #lateral coefficients
        self.p_cy1 = []  #Shape factor Cfy for lateral forces
        self.p_dy1 = []  #Lateral friction Muy
        self.p_dy3 = []  #Variation of friction Muy with squared camber
        self.p_ey1 = []  #Lateral curvature Efy at Fznom
        self.p_ky1 = []  #Maximum value of stiffness Kfy/Fznom
        self.p_hy1 = []  #Horizontal shift Shy at Fznom
        self.p_hy3 = []  #Variation of shift Shy with camber
        self.p_vy1 = []  #Vertical shift in Svy/Fz at Fznom
        self.p_vy3 = []  #Variation of shift Svy/Fz with camber
        self.r_by1 = []  #Slope factor for combined Fy reduction
        self.r_by2 = []  #Variation of slope Fy reduction with alpha
        self.r_by3 = []  #Shift term for alpha in slope Fy reduction
        self.r_cy1 = []  #Shape factor for combined Fy reduction
        self.r_ey1 = []  #Curvature factor of combined Fy
        self.r_hy1 = []  #Shift factor for combined Fy reduction
        self.r_vy1 = []  #Kappa induced side force Svyk/Muy*Fz at Fznom
        self.r_vy3 = []  #Variation of Svyk/Muy*Fz with camber
        self.r_vy4 = []  #Variation of Svyk/Muy*Fz with alpha
        self.r_vy5 = []  #Variation of Svyk/Muy*Fz with kappa
        self.r_vy6 = []  #Variation of Svyk/Muy*Fz with atan(kappa)
