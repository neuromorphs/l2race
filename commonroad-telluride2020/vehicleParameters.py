from .steeringParameters import SteeringParameters
from .longitudinalParameters import LongitudinalParameters
from .tireParameters import TireParameters

# https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass
from numba import float64, deferred_type  # import the types
from numba.experimental import jitclass


# steering_type = deferred_type()
# steering_type.define(SteeringParameters.class_type.instance_type)
# longitudinal_type = deferred_type()
# longitudinal_type.define(LongitudinalParameters.class_type.instance_type)
# tire_type = deferred_type()
# tire_type.define(TireParameters.class_type.instance_type)

spec = [
    ('l', float64),
    ('w', float64),
    # ('steering', steering_type),
    # ('longitudinal', longitudinal_type),
    ('m', float64),
    ('m_s', float64),
    ('m_uf', float64),
    ('m_ur', float64),
    ('a', float64),
    ('b', float64),
    ('I_Phi_s', float64),
    ('I_y_s', float64),
    ('I_z', float64),
    ('I_xz_s', float64),
    ('K_sf', float64),
    ('K_sdf', float64),
    ('K_sr', float64),
    ('K_sdr', float64),
    ('T_f', float64),
    ('T_r', float64),
    ('K_ras', float64),
    ('K_tsf', float64),
    ('K_tsr', float64),
    ('K_rad', float64),
    ('K_zt', float64),
    ('h_cg', float64),
    ('h_raf', float64),
    ('h_rar', float64),
    ('h_s', float64),
    ('I_uf', float64),
    ('I_ur', float64),
    ('I_y_w', float64),
    ('K_lt', float64),
    ('R_w', float64),
    ('T_sb', float64),
    ('T_se', float64),
    ('D_f', float64),
    ('D_r', float64),
    ('E_f', float64),
    ('E_r', float64),
    # ('tire', tire_type),
]


# @jitclass(spec)
class VehicleParameters():
    def __init__(self):
        # vehicle body dimensions
        self.l = 0.
        self.w = 0.

        # steering parameters
        self.steering = SteeringParameters()

        # longitudinal parameters
        self.longitudinal = LongitudinalParameters()

        # masses
        self.m = 0.
        self.m_s = 0.
        self.m_uf = 0.
        self.m_ur = 0.

        # axes distances
        self.a = 0.  # distance from spring mass center of gravity to front axle [m]  LENA
        self.b = 0.  # distance from spring mass center of gravity to rear axle [m]  LENB

        # moments of inertia of sprung mass
        self.I_Phi_s = 0.  # moment of inertia for sprung mass in roll [kg m^2]  IXS
        self.I_y_s = 0.  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
        self.I_z = 0.  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        self.I_xz_s = 0.  # moment of inertia cross product [kg m^2]  IXZ

        # suspension parameters
        self.K_sf = 0.  # suspension spring rate (front) [N/m]  KSF
        self.K_sdf = 0.  # suspension damping rate (front) [N s/m]  KSDF
        self.K_sr = 0.  # suspension spring rate (rear) [N/m]  KSR
        self.K_sdr = 0.  # suspension damping rate (rear) [N s/m]  KSDR

        # geometric parameters
        self.T_f = 0.  # track width front [m]  TRWF
        self.T_r = 0.  # track width rear [m]  TRWB
        self.K_ras = 0.  # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

        self.K_tsf = 0.  # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
        self.K_tsr = 0.  # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
        self.K_rad = 0.  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
        self.K_zt = 0.  # vertical spring rate of tire [N/m]  TSPRINGR

        self.h_cg = 0.  # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
        self.h_raf = 0.  # height of roll axis above ground (front) [m]  HRAF
        self.h_rar = 0.  # height of roll axis above ground (rear) [m]  HRAR

        self.h_s = 0.  # M_s center of gravity above ground [m]  HS

        self.I_uf = 0.  # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
        self.I_ur = 0.  # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
        self.I_y_w = 0.  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

        self.K_lt = 0.  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
        self.R_w = 0.  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

        # split of brake and engine torque
        self.T_sb = 0.
        self.T_se = 0.

        # suspension parameters
        self.D_f = 0.  # [rad/m]  DF
        self.D_r = 0.  # [rad/m]  DR
        self.E_f = 0.  # [needs conversion if nonzero]  EF
        self.E_r = 0.  # [needs conversion if nonzero]  ER

        # tire parameters
        self.tire = TireParameters()

# we use this type in other places for numba jit decorators on the car models
# vehicle_params_type=deferred_type()
# vehicle_params_type.define(VehicleParameters.class_type.instance_type) # define numba type for VehicleParameters class instance that has model parameters (it also has @jitclass)
