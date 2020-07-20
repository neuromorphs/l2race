from .steeringParameters import SteeringParameters
from .longitudinalParameters import LongitudinalParameters
from .tireParameters import TireParameters

class VehicleParameters():
    def __init__(self):
        #vehicle body dimensions
        self.l = []
        self.w = []

	#steering parameters 
        self.steering = SteeringParameters()

	#longitudinal parameters 
        self.longitudinal = LongitudinalParameters()
        
        #masses
        self.m = []
        self.m_s = []
        self.m_uf = []
        self.m_ur = []
            
        #axes distances
        self.a = []  #distance from spring mass center of gravity to front axle [m]  LENA
        self.b = []  #distance from spring mass center of gravity to rear axle [m]  LENB

        #moments of inertia of sprung mass
        self.I_Phi_s = []  #moment of inertia for sprung mass in roll [kg m^2]  IXS
        self.I_y_s = []  #moment of inertia for sprung mass in pitch [kg m^2]  IYS
        self.I_z = []  #moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        self.I_xz_s = []  #moment of inertia cross product [kg m^2]  IXZ

        #suspension parameters
        self.K_sf = []  #suspension spring rate (front) [N/m]  KSF
        self.K_sdf = []  #suspension damping rate (front) [N s/m]  KSDF
        self.K_sr = []  #suspension spring rate (rear) [N/m]  KSR
        self.K_sdr = []  #suspension damping rate (rear) [N s/m]  KSDR

        #geometric parameters
        self.T_f = []  #track width front [m]  TRWF
        self.T_r = []  #track width rear [m]  TRWB
        self.K_ras = []  #lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

        self.K_tsf = []  #auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
        self.K_tsr = []  #auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
        self.K_rad = []  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
        self.K_zt = []  # vertical spring rate of tire [N/m]  TSPRINGR

        self.h_cg = []  #center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
        self.h_raf = []  #height of roll axis above ground (front) [m]  HRAF
        self.h_rar = []  #height of roll axis above ground (rear) [m]  HRAR

        self.h_s = []  #M_s center of gravity above ground [m]  HS

        self.I_uf = []  #moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
        self.I_ur = []  #moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
        self.I_y_w = []  #wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

        self.K_lt = []  #lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
        self.R_w = []  #effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

        #split of brake and engine torque
        self.T_sb = []
        self.T_se = []

        #suspension parameters
        self.D_f = []  #[rad/m]  DF
        self.D_r = []  #[rad/m]  DR
        self.E_f = []  #[needs conversion if nonzero]  EF
        self.E_r = []  #[needs conversion if nonzero]  ER
        
        #tire parameters 
        self.tire = TireParameters()
