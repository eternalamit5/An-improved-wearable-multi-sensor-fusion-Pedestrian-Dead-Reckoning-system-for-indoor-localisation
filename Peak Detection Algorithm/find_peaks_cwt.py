from numpy import array
from scipy import signal
import numpy as np

xs = np.arange(0, np.pi, 0.05)
data = np.sin(xs)
peakind = signal.find_peaks_cwt(data, np.arange(1,10))
peakind, xs[peakind], data[peakind]([32], array([ 1.6]), array([ 0.9995736]))