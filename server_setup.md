# How to setup the l2race server

## Install l2race
Login via SSH to the ubuntu machine and go to your home folder. 

First create a conda environment:
```shell script
conda create --name l2race python=3.8
conda activate l2race
```


Setup the l2race server like this:

```shell script
git clone --recursive https://github.com/neuromorphs/l2race.git
cd l2race
pip install -e .
```

And check if it runs:
```shell script
python src/server.py
```




## Run l2race as service

Create the file start.sh in the home folder with the following content:

```txt
#!/bin/sh
# - stdout & stderr are stored to a log file
cd l2race
/home/ubuntu/miniconda3/envs/l2race/bin/python3.8 src/server.py
```


Create a service:

Create a service file called l2race.service in the directory /lib/systemd/system/
with the following content:
```txt
[Unit]
Description=L2Race Server for automatically starting after boot

[Service]
Type=simple
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/start.sh

[Install]
WantedBy=multi-user.target
Alias=l2race.service
```

and to enable the service run:
```shell script
sudo systemctl daemon-reload
```

Check if the service runs automatically after reboot:
```shell script
sudo reboot
```


To start/stop/status the service run:
```shell script
sudo service l2race start
sudo service l2race stop
sudo service l2race status
```
