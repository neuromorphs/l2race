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

    def __init__(self, car:car=None, filebase:str=DATA_FILENAME_BASE):
        self.car:car=car
        self.filebase:str=filebase
        self.filename=None
        self.file=None
        self.num_records=0
        self.first_record_written=False

    def open(self):
        if self.file:
            logger.warning('recording {} is already open, close it and open a new one'.format(self.filename))
            return
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(DATA_FOLDER_NAME):
            logger.info('creating output folder {}'.format(DATA_FOLDER_NAME))
            os.makedirs(DATA_FOLDER_NAME)

        self.filename='{}-{}.csv'.format(DATA_FILENAME_BASE, timestr)
        self.filename=os.path.join(DATA_FOLDER_NAME, self.filename)
        if self.car==None:
            raise('car is None, cannot record data')

        try:
            self.file=open(self.filename,'w')
            print(self.car.car_state.get_record_headers(), file=self.file)
            atexit.register(self.close)
            self.num_records=0
            self.first_record_written=False
        except Exception as ex:
            self.file=None
            logger.warning('{}: could not open {} for recording data'.format(ex, self.filename))
            raise RuntimeError(ex)

    def close(self):
        if self.file:
            logger.info('closing recording {} with {} records'.format(self.filename, self.num_records))
            self.file.close()
            self.file=None

    def write_sample(self):
        if self.file==None:
            logger.warning('there is no output file open to record to')
            return
        print(self.car.car_state.get_record_csvrow(), file=self.file)
        self.first_record_written=True
        self.num_records+=1

        pass

