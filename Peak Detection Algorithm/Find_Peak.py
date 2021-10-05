import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


data = pd.read_excel (r'..\Flex Sensor\FlexSensorWalkingData.xlsx')
#data = pd.read_excel (r'f:\MSC Courses\Master Thesis BIBA\flex sensor feet data test1.xlsx')
df = pd.DataFrame(data, columns= ['Analog value'])

x = -1*df['Analog value'].values

# x = np.sin(2*np.pi*(2**np.linspace(2,10,1000))*np.arange(1000)/48000) + np.random.normal(0, 1, 1000) * 0.15
peaks, _ = find_peaks(x, distance=10)
peaks2, _ = find_peaks(x, prominence=18)      # BEST! value=10 is working good
peaks3, _ = find_peaks(x, width=2)
peaks4, _ = find_peaks(x, threshold=5)     # Required vertical distance to its direct neighbouring samples, pretty useless
plt.subplot(2, 2, 1)
#plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
#plt.subplot(2, 2, 2)
plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
#plt.subplot(2, 2, 3)
#plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
#plt.subplot(2, 2, 4)
#plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
plt.show()