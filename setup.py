from setuptools import setup, setuptools
from Cython.Build import cythonize
# run with
# python setup.py build_ext --inplace

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="l2race-tellurideneuromorphs", # Replace with your own username
    version="0.0.1",
    author="Marcin Paluch, Antonio Rios, Chang Gao, Tobi Delbruck",
    author_email="tobi@ini.uzh.ch",
    description="L2RACE challenge Telluride Neuromorphic workshop",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuromorphs/l2race",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

setup(
    ext_modules = cythonize([
        "commonroad/vehicleDynamics_MB.pyx",
        "commonroad/vehicleDynamics_ST.pyx",
        "commonroad/tireModel.pyx",
        # "commonroad/longitudinalParameters.py",
        "commonroad/accelerationConstraints.pyx",
        "commonroad/steeringConstraints.pyx",
        # "commonroad/tireParameters.py",
        # "commonroad/vehicleParameters.py",
        # "commonroad/steeringParameters.py",
        "commonroad/unitConversions/unitConversion.pyx",
        # "commonroad/parameters_vehicle1.py",
        # "commonroad/parameters_vehicle2.py",
        # "commonroad/parameters_vehicle3.py",
        ], compiler_directives={'language_level' : "3"})
)