# Import default_timer to compute durations
from timeit import default_timer as timer
import numpy as np
import scipy as sp
import pandas as pd
from glob import glob
from IPython.display import display
import matplotlib.pyplot as plt
from typing import Optional
import logging
import sys

# logging setup
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

start_time = timer()


class MeasurementExtractor:
    def __init__(self, measurement_file: str, column_labels: list) -> None:
        self._measurement_file: str = measurement_file
        self._column_labels: list = column_labels
        _,self._experiment_id, _, self._user_id, activity_str = self.measurement_filename.split('_')

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, id: str) -> None:
        self._experiment_id = id

    @property
    def user_id(self) -> str:
        return self._user_id

    @user_id.setter
    def user_id(self, id: str) -> None:
        self._user_id = id

    @property
    def measurement_column(self) -> str:
        return self._column_labels

    @measurement_column.setter
    def measurement_column(self, column_names) -> None:
        self._column_labels = column_names

    @property
    def measurement_filename(self) -> str:
        return self._measurement_file

    @staticmethod
    def _read_data(filepath: str, columns) -> pd.DataFrame:
        try:
            file = open(filepath, 'r')
        except FileNotFoundError as e:
            logger.error(f"File Not found: filename {filepath}")
            exit()

        labels = list()

        # Store each row in a list ,convert its list elements to int type
        for entry in file:
            labels.append([element for element in entry.split(',')])

        return pd.DataFrame(data=np.array(labels, dtype=float), columns=columns)

    def read_measurement(self) -> pd.DataFrame:
        return self._read_data(filepath=self.measurement_filename, columns=self._column_labels)


