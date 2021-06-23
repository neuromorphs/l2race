"""
calculates and adds sin and cos variables to all csv files in folder

usage:
python add_trig.py folder_name

folder_name needs to be relative to l2race root directory
"""

from sys import argv
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data_dir = argv[1:][0]

    full_path = Path.cwd().joinpath(data_dir)

    file_list = [x for x in full_path.glob('*.csv')]

    for filename in file_list:
        data = pd.read_csv(filename, comment='#')
        data['body_angle_sin'] = np.sin(np.deg2rad(data['body_angle_deg']))
        data['body_angle_cos'] = np.cos(np.deg2rad(data['body_angle_deg']))
        data.to_csv(filename, index=False)
