import datetime
import pandas as pd
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
import matplotlib.pyplot as plt


class FIR:
    def __init__(self, signal: np.ndarray, filter_type="lowpass", cutoff_freq_hz=5, sample_rate_hz=1,
                 windowing="kaiser", transition_band_width_hz=5, stop_band_attenuation_db=100):
        self._signal = np.array([float(i) for i in signal.tolist()])
        self._filter_type = filter_type
        self._cutoff_freq_hz = cutoff_freq_hz
        self._sample_rate_hz = sample_rate_hz
        self._sample_size = self._signal.size
        self._windowing = windowing
        self._transition_band_width_hz = transition_band_width_hz
        self._stop_band_attenuation_db = stop_band_attenuation_db

    @property
    def filter_type(self) -> str:
        return self._filter_type

    @filter_type.setter
    def filter_type(self, filtertype: str = 'lowpass'):
        self._filter_type = filtertype

    @property
    def cutoff_freq_hz(self):
        return self._cutoff_freq_hz

    @cutoff_freq_hz.setter
    def cutoff_freq_hz(self, cutoff_freq_hz):
        self._cutoff_freq_hz = cutoff_freq_hz

    def filter(self):
        nyq_rate = self._sample_rate_hz / 2.0
        N, beta = kaiserord(self._stop_band_attenuation_db, self._transition_band_width_hz / nyq_rate)
        taps = firwin(N, self._cutoff_freq_hz / nyq_rate, window=('kaiser', beta))
        return lfilter(taps, 1.0, self._signal)

    def show(self, title=''):
        sample_seq = np.arange(start=0, stop=self._sample_size, step=1)
        fig, ax = plt.subplots()
        ax.plot(sample_seq, self._signal, 'r-', label='raw signal')
        ax.plot(sample_seq, self.filter(), 'g-', label='filtered signal')
        plt.title(title)
        plt.legend()
        plt.show()
