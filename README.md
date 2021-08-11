# Learning to Race (l2race)
 
Simulation of racecar from eye of god view. User algorithms must learn from only real time data to drive the car as quickly as possible around the track. Data can be collected by human driving or by developing a basic controller and using it to bootstrap more powerful controllers.

The key aims are to learn a controller from limited 'real world' data and to use it for optimal control.

![aminated gif](media/videos/first_2_car_race_marcin_chang_1.gif)

 - [L2RACE challenge page](https://sites.google.com/view/telluride2020/challenges/l2race?authuser=0) on workshop site
 - The [L2RACE google driver folder](https://drive.google.com/drive/folders/1IJmfvKY2n24PQTGxc9Ek4ApufMVISC1C?usp=sharing).
 - In particular, [L2RACE introductory slides](https://docs.google.com/presentation/d/1nvmZqHNQrUKONi_r6YIpepk4Ie-LkIc1N1GxrF_Ehv0/edit?usp=sharing).
 

## Requirements

 - Windows, linux, macOS all seem to work
 - anaconda or miniconda https://www.anaconda.com/products/individual
 - Use Python 3.7.x (for pygame 2.0, using prebuilt python wheel archive)
 - We use pycharm for development, which includes some useful launchers to start local server, client, client to our remote model server. https://www.jetbrains.com/pycharm/download . You can use the community edition.


## Setup
The most straight forward way to install l2race is to use the terminal.
Conda is your friend!

L2Race uses git submodules, so when you clone the project make sure you also clone them:
```shell script
git clone --recursive https://github.com/neuromorphs/l2race.git
cd l2race
```


# Installation

The -e and develop options install referring to your working code, so your edits take effect on console script invocations.

Make sure miniconda or conda is installed. Create a new python environment and activate it. Finally install the requirements.

```shell script
conda create --name l2race python=3.8
conda activate l2race
python setup.py develop # develop option let's you run your install directly from source code and your changes to source code take effect immediately
```

Now you can run the l2race client.

In one terminal
```shell script
$ l2race-server
pygame 2.0.1 (SDL 2.0.14, Python 3.8.5)
Hello from the pygame community. https://www.pygame.org/contribute.html
[38;21m2021-07-17 23:38:43,409 - src.server - INFO - waiting on <socket.socket fd=352, family=Address
Family.AF_INET, type=SocketKind.SOCK_DGRAM, proto=0, laddr=('0.0.0.0', 50000)> (server.py:314)[0m
[38;21m2021-07-17 23:38:43,928 - src.server - INFO - received cmd "ping" with payload "None" from ('1
27.0.0.1', 54586) (server.py:442)[0m
[38;21m2021-07-17 23:38:43,932 - src.server - INFO - received cmd "add_car" with payload "('empty', '
LAPTOP-SS5VU6HO:tobid-CW')" from ('127.0.0.1', 54586) (server.py:442)[0m
....
```

In another terminal
```shell script
$ l2race-client
pygame 2.0.1 (SDL 2.0.14, Python 3.8.5)
Hello from the pygame community. https://www.pygame.org/contribute.html
...
```

### Troubleshooting
#### Pip

You have to install the requirements for the l2race environment using its own pip.
If the installation of the requirements does not work, make sure you are using the conda pip in your conda environment:
```shell script
where pip
C:\Users\tobid\anaconda3\envs\l2race\Scripts\pip.exe
```
#### Git Submodules
If you use an old version of git the submodules might not be loaded correctly. If there is an error like "module not found: VehicleModels", this is probably the case.
Check the folder "commonroad-vehicle-models". Is it empty?
Run:
```shell script
git submodule update --init --recursive
```
Check again and if it is no longer empty, the submodules have been loaded correctly.


#### Pygame
The necessary pygame 2.0 seems to install into windows and linux and macOS directly with pip now.

If you still have problems, you can see the pygame 2.0-dev10 (needed for python 3.7) wheels at https://www.piwheels.org/project/pygame/ or https://github.com/pygame/pygame/releases.  (A _wheel_ is a kind of archi8ve of python stuff with all dependencies; they are named according to the platform and OS). Then use pip install wheel-file. Download the wheel for pygame 2.0 for python 3.7 and your OS.

#### Pytorch
If you want to use our RNN models you additionally need to install Pytorch. 
To install a working version of Pytorch you need not only specify your OS but also CUDA support for your GPU. 
We therefore recommend that you check the right installation command directly at the [Pytorch official webpage](https://pytorch.org).


## pycharm
You should be able to open l2race project from pycharm directly, since l2race includes the jetbrains/pycharm .idea folder.
Once in pycharm, if you have already setup the l2race conda environment, then pycharm should find it. 
If not, set up the conda enviroment in pycharm using Project settings/Project interpreter and point to your l2race conda environment:

![pycharm setup](media/pycharm_env.png)

# Running l2race

l2race uses a client-server architecture.
The client draws the racetrack and car and accepts input from keyboard or xbox joystick controller or your software agent.
The server computes the car dynamics model in response to your command input and returns the car state to the client.

From root of l2race, start the server and client from separate terminals (or from pycharm; see below).

### Start the server
If you want to run the server on your local machine, do it like this:

```shell script
(l2race) python -m src.client

```
### Start client (typical remote use case)
```shell script
(l2race) python -m src.server
```



### Server options

````shell script
python  src/server.py -h
pygame 2.0.1 (SDL 2.0.14, Python 3.8.6)
Hello from the pygame community. https://www.pygame.org/contribute.html
usage: server.py [-h] [--allow_off_track] [--log LOG] [--port PORT]

l2race client: run this if you are a racer.

optional arguments:
  -h, --help         show this help message and exit

Server arguments::
  --allow_off_track  ignore when car goes off track (for testing car dynamics
                     more easily) (default: False)
  --log LOG          Set logging level. From most to least verbose, choices
                     are "DEBUG", "INFO", "WARNING". (default: INFO)
  --port PORT        Server port address for initiating connections from
                     clients. (default: 50000)

Run with no arguments to open dialog for server IP
````

## Client options
````console

D:\envs\l2race\python.exe src/client.py -h
pygame 2.0.1 (SDL 2.0.14, Python 3.8.5)
Hello from the pygame community. https://www.pygame.org/contribute.html
usage: client.py [-h] [--host HOST] [--port PORT] [--timeout_s TIMEOUT_S]
                 [--fps FPS] [--joystick JOYSTICK] [--record [RECORD]]
                 [--replay [REPLAY]] [--autodrive AUTODRIVE AUTODRIVE]
                 [--carmodel CARMODEL CARMODEL] [--lidar [LIDAR]]
                 [--track_name {empty,oval,oval_easy,Sebring,track_1,track_2,track_3,track_4,track_5,track_6,,dialog,choose}]
                 [--car_name CAR_NAME] [--spectate] [--log LOG]

l2race client: run this if you are a racer.

optional arguments:
  -h, --help            show this help message and exit
  --log LOG             Set logging level. From most to least verbose, choices
                        are "DEBUG", "INFO", "WARNING". (default: INFO)

Model server connection options::
  --host HOST           IP address or DNS name of model server. (default:
                        localhost)
  --port PORT           Server port address for initiating connections.
                        (default: 50000)
  --timeout_s TIMEOUT_S
                        Socket timeout in seconds for communication with model
                        server. (default: 1)

Interface arguments::
  --fps FPS             Frame rate on client side (server always sets time to
                        real time). (default: 20)
  --joystick JOYSTICK   Desired joystick number, starting with 0. (default: 0)

Output/Replay options::
  --record [RECORD]     Record data to date-stamped filename with optional
                        <note>, e.g. --record will write datestamped files
                        named 'l2race-<track_name>-<car_name>-<note>-TTT.csv'
                        in folder 'data, where note is optional note and TTT
                        is a date/timestamp'. (default: None)
  --replay [REPLAY]     Replay one or more CSV recordings. If 'last' or no
                        file is supplied, play the most recent recording in
                        the 'data' folder. (default: None)

Control/Modeling arguments::
  --autodrive AUTODRIVE AUTODRIVE
                        The autodrive module and class to be run when
                        autodrive is enabled on controller. Pass it the module
                        (i.e. folder.file without .py) and the class within
                        the file. (default:
                        ['src.controllers.neural_mpc_controller',
                        'neural_mpc_controller'])
  --carmodel CARMODEL CARMODEL
                        Your client car module and class and class to be run
                        as ghost car when model evaluation is enabled on
                        controller. Pass it the module (i.e. folder.file
                        without .py) and the class within the file. (default:
                        ['models.models', 'linear_extrapolation_model'])

Sensor arguments::
  --lidar [LIDAR]       Draw the point at which car would hit the track edge
                        if moving on a straight line. The numerical value
                        gives precision in pixels with which this point is
                        found. (default: None)

Track car/spectate options::
  --track_name {empty,oval,oval_easy,Sebring,track_1,track_2,track_3,track_4,track_5,track_6,,dialog,choose}
                        Name of track (or empty string or 'none' or 'choose'
                        or 'dialog' to show dialog). Available tracks are in
                        the './media/tracks/' folder, defined by
                        globals.TRACKS_FOLDER. (default: dialog)
  --car_name CAR_NAME   Name of this car (last 2 letters are randomly chosen
                        each time). (default: None)
  --spectate            Just be a spectator on the cars on the track.
                        (default: False)

````

### Joystick and keyboard
 - Help for each device is printed on startup. For keyboard, you can type h anytime to see the keys help in console.
 - You need to focus on the pygame window for either input to work.

# Development

## pycharm

l2race includes pycharm _.idea_ files that have many useful run configurations already set up.

# Recording data

The _--record_ option automatically records a .csv file with timestamped filename to the _data_ folder. This file has the time, commnands, and car state.

# Drawing and generating tracks

Track information used by _l2race_ is generated by the scripts in _Track_Preparation_ _draw_tracks.py_ and _create_track_info.py_.

To update the track information files (npy and png), run the create_track_info.py script in TrackPreparation

These scripts must be run from their folder. 

Tracks start from monochrome template most easily. See T*rack_Preparation/track_templates* for examples.

Starting from the track templates in _Track_Preparation/tracks_templates_, they generate the files in the _media/tracks_ folder named (for track named _track_) _track.png_, _track_info.npy_, and _track_map.npy_ needed by _track.py_.

Sample runs follow.

## draw_tracks.py

    "C:\Program Files\JetBrains\PyCharm 2020.1.4\bin\runnerw64.exe" C:\Users\tobid\anaconda3\envs\l2race\python.exe "F:/tobi/Dropbox (Personal)/GitHub/neuromorphs/l2race/Track_Preparation/draw_track.py"
    Processing track Sebring starting from template ./tracks_templates/Sebring.png
    saving ../media/tracks/Sebring.png
    saving grayscale frame ./tracks_gray/Sebring_G.png
    saving starting position frame ./tracks_start/Sebring_start.png
    Processing track oval starting from template ./tracks_templates/oval.png
    saving ../media/tracks/oval.png
    saving grayscale frame ./tracks_gray/oval_G.png

## write_track_info_files.py

    "C:\Program Files\JetBrains\PyCharm 2020.1.4\bin\runnerw64.exe" C:\Users\tobid\anaconda3\envs\l2race\python.exe "F:/tobi/Dropbox (Personal)/GitHub/neuromorphs/l2race/Track_Preparation/get_track_info.py"
    Now processing: Sebring
    loading gray scale track image ./tracks_gray/Sebring_G.png
    You have more than 2 contours in Sebring
    I change the direction of Sebring
    saving ../media/tracks/Sebring_map.npy and ../media/tracks/Sebring_info.npy
    Now processing: oval
    loading gray scale track image ./tracks_gray/oval_G.png
    I change the direction of oval
    saving ../media/tracks/oval_map.npy and ../media/tracks/oval_info.npy


