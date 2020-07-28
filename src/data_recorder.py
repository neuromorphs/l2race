# records data from l2race car
import os
from collections import OrderedDict

from car import car

import pandas as pd
import datetime
from src.globals import DATA_FILENAME_BASE, DATA_FOLDER_NAME
from src.my_logger import my_logger
import atexit

logger = my_logger(__name__)


class data_recorder:

    def __init__(self, car:car=None):
        self.car:car=car
        self.filename:str=None
        self.file=None
        self.writer=None
        self.columns=[]
        self.dataframe={}
        self.adddict('time')
        self.adddict('steering')
        self.adddict('throttle')
        self.adddict('brake')
        self.adddict('pos.x','position_m.x')
        self.adddict('pos.y','position_m.y')
        self.adddict('vel.x','velocity_m_per_sec.x')
        self.fielddict=pd.DataFrame(self.d)

        # self.fieldnames=OrderedDict(['time', 'cmd.steering', 'cmd.throttle', 'cmd.brake', 'state.pox.x', 'state.pos.y', 'state.vel.x', 'state.vel.y', 'state.speed', 'state.accel', 'state.body_angle', 'state.yaw_rate', 'state.drift_angle']['time', 'cmd.steering', 'cmd.throttle', 'cmd.brake', 'state.pox.x', 'state.pos.y', 'state.vel.x', 'state.vel.y', 'state.speed', 'state.accel', 'state.body_angle', 'state.yaw_rate', 'state.drift_angle']

    def adddict(self,name):
        self.columns.append(name)
        self.d[name]=name

    def adddict(self,name,field):
        self.d[name]=field

    def open(self):
        '''open a new automatically-named output file'''
        filename='{}-{}-{}.csv'.format(DATA_FILENAME_BASE, DATA_FILENAME_BASE, str(datetime.date(), str(datetime.time())))
        os.path.join(DATA_FOLDER_NAME, filename)
        self.open(filename)

    def open(self, filename):
        self.filename=filename
        if self.car==None:
            raise('car is None, cannot record data')

        try:
            self.file=open(self.filename,'w')
            self.writer=csv.writer(self.file)
            self.writer.writerow(self.fielddict)
            atexit.register(self.close())
        except:
            logger.warning('could not open {} for recording data'.format(self.filename))

    def close(self):
        if self.writer:
            logger.info('closing recording {}'.format(self.filename))
            self.writer.close()

    def record(self):
        #todo write the actual state according to fields, must be some way to automate it from list of fields
        row=[]
        for k,v in self.fielddict:
            val=getattr(self.car.car_state,v)
            row.append(val)
        self.writer.writerow(row)

        pass

