from setuptools import setup
from Cython.Build import cythonize
# run with
# python setup.py build_ext --inplace
setup(
    ext_modules = cythonize([
        "commonroad/vehicleDynamics_MB.py",
        "commonroad/vehicleDynamics_ST.py",
        "commonroad/vehicleDynamics_KS.py",
        "commonroad/tireModel.py",
        "commonroad/longitudinalParameters.py",
        "commonroad/accelerationConstraints.py",
        "commonroad/steeringConstraints.py",
        "commonroad/tireParameters.py",
        "commonroad/vehicleParameters.py",
        "commonroad/steeringParameters.py",
        "commonroad/unitConversions/unitConversion.py",
        "commonroad/parameters_vehicle1.py",
        "commonroad/parameters_vehicle2.py",
        "commonroad/parameters_vehicle3.py",
        ])
)