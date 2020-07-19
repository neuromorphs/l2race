# Learning to Race (l2race)
 
Simulation of racecar from eye of god view. User code must drive the car as quickly as possible around the track.

## Requirements

Python 3.7.x (for pygame, using prebuilt python wheel arxiv)

For pygame, see wheels at https://www.piwheels.org/project/pygame/ or https://github.com/pygame/pygame/releases . Then use pip install wheel-file. Download the wheel for pygame 2.0 for python 3.7 and your OS.

requirements.txt was built automatically using https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt
```shell script
pip install pipreqs
pipreqs --force .
```
You can build the conda env l2race using
```shell script
conda env create -f environment.yml
```

See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

For server, it is not ideal, but for now just clone the commonroad repo at the same level as l2race, i.e. side by side in same containing folder. I.e. make a folder, then have l2race and commonroad both in that folder. Then l2race adds the path to that folder when it starts up.

It is not ideal because it means that you cannot navigate to those classes easily (pycharm thinks they cannot be found). The solution would be to add the commonroad folder to PYTHONPATH, but this is also a hack.