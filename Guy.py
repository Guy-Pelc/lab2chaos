#This script plots graph, maximum and minimum points of given file.
#Usage: Edit file and folder address, adjust N as needed

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema


FOLDER_ADDRESS = "week 2"
FILE_ADDRESS = "100mV"
N=100 # number of points to be checked before and after


df = pd.read_csv(FOLDER_ADDRESS+"/"+FILE_ADDRESS + ".csv")

df['data'] = df.iloc[:,4]
# Find local peaks
df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal, order=N)[0]]['data']
df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal, order=N)[0]]['data']

# Plot results
plt.scatter(df.index, df['min'], c='r')
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['data'])
plt.show()