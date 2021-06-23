# records data from l2race car
import os
from collections import OrderedDict

from src.car import car

from src.globals import DATA_FILENAME_BASE, DATA_FOLDER_NAME
from src.l2race_utils import my_logger
import atexit

logger = my_logger(__name__)


class data_recorder:

    def __init__(self, car:car,  note:str=None, filebase:str=DATA_FILENAME_BASE):
        self.car:car=car
        self.filebase:str=filebase
        self.filename=None
        self.file=None
        self.num_records=0
        self.first_record_written=False
        self.note=note

    def open_new_recording(self)->None:
        """
        Creates a new recording if it is not already open.

        :return: None
        :raises RuntimeError if it cannot open the recording
        """
        if self.file:
            logger.warning('recording {} is already open, close it and open a new one'.format(self.filename))
            return

        import time
        timestr = time.strftime("%Y%m%d-%H%M%S") # e.g. '20200819-1601'
        if not os.path.exists(DATA_FOLDER_NAME):
            logger.info('creating output folder {}'.format(DATA_FOLDER_NAME))
            os.makedirs(DATA_FOLDER_NAME)

        namestring=str.split(self.car.name(),'-')[0]
        if self.note!='' and not self.note is None:
            self.filename='{}-{}-{}-{}-{}.csv'.format(DATA_FILENAME_BASE,  namestring, self.car.track.name, self.note, timestr)
        else:
            self.filename='{}-{}-{}-{}.csv'.format(DATA_FILENAME_BASE, namestring, self.car.track.name, timestr)
        self.filename=os.path.join(DATA_FOLDER_NAME, self.filename)

        try:
            self.file=open(self.filename,'w')
            print(self.car.car_state.get_csv_file_header(self.car), file=self.file)
            atexit.register(self.close_recording)
            self.num_records=0
            self.first_record_written=False
            logger.info('created new recording {}'.format(self.filename))
        except Exception as ex:
            self.file=None
            logger.warning('{}: could not open {} for recording data'.format(ex, self.filename))
            raise RuntimeError(ex)

    def close_recording(self):
        if self.file:
            logger.info('closing recording {} with {} records'.format(self.filename, self.num_records))
            self.file.close()
            self.file=None
        else:
            logger.warning('no recording {} to close, maybe never opened?'.format(self.filename))

    def write_sample(self):
        if self.file is None:
            logger.warning('there is no output file open to record to')
            return
        print(self.car.car_state.get_record_csvrow(), file=self.file)
        self.first_record_written=True
        self.num_records+=1

        pass

