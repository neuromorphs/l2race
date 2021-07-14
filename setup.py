from setuptools import setup, find_packages
# from Cython.Build import cythonize
import sys
sys.path.insert(0, 'src/')

# run with
# python setup.py build_ext --inplace

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="l2race", # Replace with your own username
    version="2.0.1",
    author="Tobi Delbruck, Marcin Paluch, Antonio Rios",
    author_email="tobi@ini.uzh.ch,marcin.paluch1994@gmail.com,arios@us.es",
    description="L2RACE challenge for Telluride Neuromorphic workshop",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuromorphs/l2race",
    packages=find_packages(),
    # scripts=['client.py','server.py'],
    entry_points = {
        'console_scripts': ['l2race-client=src.client:main', 'l2race-server=src.server:main']
    },
    install_requires=[
        'svgpathtools',
        'pygame==2.0.1',
        'matplotlib',
        'numpy==1.19.2',
        'argparse',
        'argcomplete',
        'opencv-python',
        # numba # not using now
        # 'cython', # cython needs to be compiled using setup.py
        'scikit-learn>=0.24.1',
        'scipy',
        # 'scikit-learn-extras',
        'upnpy',
        'pandas',
        'geos', # needed since shapely does not install geos_c.dll needed, solved by geos install
        'shapely',
        'tensorflow>=2.5',
        # 'pysindy',
        'torch', # go to https://pytorch.org/get-started/locally/ for correct pip or conda command generator
        'IPython',
        'cvxpy',
        'easygui'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # ext_modules = cythonize([
    #     "commonroad/vehicleDynamics_MB.pyx",
    #     "commonroad/vehicleDynamics_ST.pyx",
    #     "commonroad/tireModel.pyx",
    #     # "commonroad/longitudinalParameters.py",
    #     "commonroad/accelerationConstraints.pyx",
    #     "commonroad/steeringConstraints.pyx",
    #     # "commonroad/tireParameters.py",
    #     # "commonroad/vehicleParameters.py",
    #     # "commonroad/steeringParameters.py",
    #     "commonroad/unitConversions/unitConversion.pyx",
    #     # "commonroad/parameters_vehicle1.py",
    #     # "commonroad/parameters_vehicle2.py",
    #     # "commonroad/parameters_vehicle3.py",
    # ], compiler_directives={'language_level' : "3"})
)

# needs to be run on server after touching any of the pyx files with "python setup.py build_ext --inplace"
