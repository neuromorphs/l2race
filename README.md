# Learning to Race (l2race)
 
Simulation of racecar from eye of god view. User algorithms must learn from only real time data to drive the car as quickly as possible around the track. Data can be collected by human driving or by developing a basic controller and using it to bootstrap more powerful controllers.

The key points are to learn a controller from limited 'real world' data and to use it for optimal control.

![aminated gif](media/SampleRun_2020-07-25_075130_2.gif)

 - [L2RACE challenge page](https://sites.google.com/view/telluride2020/challenges/l2race?authuser=0) on workshop site
 - The [L2RACE google driver folder](https://drive.google.com/drive/folders/1IJmfvKY2n24PQTGxc9Ek4ApufMVISC1C?usp=sharing).
 - In particular, [L2RACE introductory slides](https://docs.google.com/presentation/d/1nvmZqHNQrUKONi_r6YIpepk4Ie-LkIc1N1GxrF_Ehv0/edit?usp=sharing).
 

## Requirements

 - Windows, linux, macOS all seem to work
 - anaconda or miniconda https://www.anaconda.com/products/individual
 - Use Python 3.7.x (for pygame 2.0, using prebuilt python wheel archive)

Conda is your friend! Make a new environment to work in l2race.
You can install the requirements in this environment using its own pip.
You can try to build the entire conda env l2race using

```shell script
conda env create -f environment.yml
```

If this does not work for some reason (some libraries are still not available from conda repos), then you can also use pip to install the requirements into your conda environment.
Make a new environment (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)
```shell script
conda create --name l2race python=3.7
```
Activate it:
```shell script
conda activate l2race
```
Make sure you are using the conda pip in your conda environment:
```shell script
where pip
C:\Users\tobid\anaconda3\envs\l2race\Scripts\pip.exe
```
Install the requiremepnts:

```shell script
pip install -r requirements.txt
``` 
### pygame
For pygame 2.0-dev10 (needed for python 3.7), see wheels at https://www.piwheels.org/project/pygame/ or https://github.com/pygame/pygame/releases . Then use pip install wheel-file. Download the wheel for pygame 2.0 for python 3.7 and your OS.

_requirements.txt_ was built automatically using https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt
```shell script
pip install pipreqs
pipreqs --force .
```

# Running l2race

l2race uses client-server. 

The client draws the racetrack and car and accepts input from keyboard or xbox joystick controller or your software agent.

The server computes the car dynamics model in response to your command input and returns the car state to the client.

From root of l2race, start the server and client from separate terminals (or from pycharm; see below).

### start client
```shell script
(l2race) F:\tobi\Dropbox (Personal)\GitHub\neuromorphs\l2race>python -m server
pygame 2.0.0.dev10 (SDL 2.0.12, python 3.7.7)
Hello from the pygame community. https://www.pygame.org/contribute.html
WARNING:commonroad.vehicleDynamics_MB:check_cython: F:\tobi\Dropbox (Personal)\GitHub\neuromorphs\l2race\commonroad\vehicleDynamics_MB.py is still just a slowly interpreted script.
WARNING:__main__:Gooey GUI builder not available, will use command line arguments.
Install with "pip install Gooey". See README
WARNING:__main__:Gooey GUI not available, using command line arguments.
You can try to install with "pip install Gooey"
INFO:__main__:waiting on <socket.socket fd=1512, family=AddressFamily.AF_INET, type=SocketKind.SOCK_DGRAM, proto=0, laddr=('0.0.0.0', 50000)>
```

Start the server:

```shell script
(l2race) F:\tobi\Dropbox (Personal)\GitHub\neuromorphs\l2race>python -m server
pygame 2.0.0.dev10 (SDL 2.0.12, python 3.7.7)
Hello from the pygame community. https://www.pygame.org/contribute.html
WARNING:commonroad.vehicleDynamics_MB:check_cython: F:\tobi\Dropbox (Personal)\GitHub\neuromorphs\l2race\commonroad\vehicleDynamics_MB.py is still just a slowly interpreted script.
WARNING:__main__:Gooey GUI builder not available, will use command line arguments.
Install with "pip install Gooey". See README
WARNING:__main__:Gooey GUI not available, using command line arguments.
You can try to install with "pip install Gooey"
```

Don't worry about missing Gooey; install it if you want to have a GUI pop up to launch the client and server.

It should not start running the client and you should see this:

![screenshot](media/oval_track_screenshot.png)

## client options
````shell script
(l2race) F:\tobi\Dropbox (Personal)\GitHub\neuromorphs\l2race>python -m client -h
usage: client.py [-h] [--host HOST] [--port PORT] [--fps FPS]
                 [--joystick JOYSTICK] [--record] [--car_name CAR_NAME]
                 [--track_name TRACK_NAME] [--spectate]
                 [--timeout_s TIMEOUT_S] [--log LOG]

l2race client: run this if you are a racer.

optional arguments:
  -h, --help            show this help message and exit

Client arguments::
  --host HOST           IP address or DNS name of model server. (default:
                        localhost)
  --port PORT           Server port address for initiating connections.
                        (default: 50000)
  --fps FPS             Frame rate on client side (server always sets time to
                        real time). (default: 30)
  --joystick JOYSTICK   Desired joystick number, starting with 0. (default: 0)
  --record              record data to date-stamped filename, e.g. --record
                        will write datestamped files named 'l2race-XXX.csv' in
                        folder 'data, where XXX is a date/timestamp'.
                        (default: False)
  --car_name CAR_NAME   Name of this car. (default: tobi-joule-amd:tobid)
  --track_name TRACK_NAME
                        Name of track. Available tracks are in the
                        'media/tracks' folder. Available tracks are ['oval',
                        'oval_easy', 'Sebri', 'track_1', 'track_2', 'track_3',
                        'track_4', 'track_5', 'track_6'] (default: oval_easy)
  --spectate            Just be a spectator on the cars on the track.
                        (default: False)
  --timeout_s TIMEOUT_S
                        Socket timeout in seconds for communication with model
                        server. (default: 1)
  --log LOG             Set logging level. From most to least verbose, choices
                        are "DEBUG", "INFO", "WARNING". (default: INFO)

Run with no arguments to open dialog for server IP

````

### joystick and keyboard
 - Help for each device is printed on startup. For keyboard, you can type h anytime to see the keys help in console.
 - You need to focus on the pygame window for either input to work.

# Development

## pycharm

l2race includes pycharm _.idea_ files that have many useful run configurations already set up.

## Recording data

The _--record_ option automatically records a .csv file with timestamped filename to the _data_ folder. This file has the time, commnands, and car state.
