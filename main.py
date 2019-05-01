#This script plots graph, maximum and minimum points of given file.
#Usage: Edit file and folder address, adjust N as needed

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, find_peaks_cwt, find_peaks, ricker


FOLDER_ADDRESS = "week 2"
FILE_ADDRESS = "100mV"
MAX_ROWS = 1000000
ORDER = 500

TIME_COL_NAME = 'time'
GEN_VOL_COL_NAME = 'gen_v'
DIODE_VOL_COL_NAME = 'diode_v'
PREV_MAX_COL_NAME = 'prev_max'

# Maxima filtering parameters
MAXIMA_TIMEDIFF_BINS = 20
MAXIMA_TIMEDIFF_QUANTILE_BAR = 0.2


def extract_raw_data(filename):
    """
    Takes a data file and returns a useable data frame from it.
    :return: a data frame with columns: 'time', 'gen_v' which is the voltage on the generator
    and 'diode_v' which is the voltage on the diode.
    """
    raw_data = pd.read_csv(filename)
    data = pd.DataFrame({TIME_COL_NAME: raw_data.iloc[:, 3],
                         GEN_VOL_COL_NAME: raw_data.iloc[:, 4],
                         DIODE_VOL_COL_NAME: raw_data.iloc[:, 10]})
    subsample_rate = np.ceil(data.shape[0] / MAX_ROWS)
    data = data.iloc[::subsample_rate, :]
    return data


def plot_raw_data(df):
    plt.plot(df[TIME_COL_NAME], df[DIODE_VOL_COL_NAME])
    plt.show()


# Deprecated for now
"""
def filter_maxima(maxima_rows):
    bins = np.arange(0, MAXIMA_TIMEDIFF_BINS)
    timediffs = maxima_rows[TIME_COL_NAME].diff().fillna(0)
    timediffs = pd.cut(timediffs, MAXIMA_TIMEDIFF_BINS, labels=bins).cat.codes.sort_index()
    plt.hist(timediffs, MAXIMA_TIMEDIFF_BINS, rwidth=0.5)
    plt.show()
    print(timediffs)
    timediff_mode = timediffs[timediffs > int(MAXIMA_TIMEDIFF_BINS*MAXIMA_TIMEDIFF_QUANTILE_BAR)].mode()[0]
    print(timediff_mode)
    tolerance = int(0.2 * MAXIMA_TIMEDIFF_BINS)
    maxima_rows = maxima_rows[(timediffs <= (timediff_mode + tolerance)) & (timediffs >= (timediff_mode - tolerance))]
    return maxima_rows
"""


def display_data(data ,maxima_rows):
    disp_df = data[(data[TIME_COL_NAME] > 0) & (data[TIME_COL_NAME] < 0.002)]
    disp_maxima = maxima_rows[(maxima_rows[TIME_COL_NAME] > 0) & (maxima_rows[TIME_COL_NAME] < 0.002)]
    plt.plot(disp_df[TIME_COL_NAME], disp_df[DIODE_VOL_COL_NAME], linewidth=0.5)
    plt.scatter(disp_maxima[TIME_COL_NAME], disp_maxima[DIODE_VOL_COL_NAME], s=4, c='red')
    plt.suptitle("Data Segment With Detected Maxima")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.show()
    timediffs = maxima_rows[TIME_COL_NAME].diff().fillna(0)
    plt.hist(timediffs, rwidth=0.5)
    plt.suptitle("Histogram of Time Difference Between Maxima")
    plt.xlabel("Time Difference Between Consecutive Peaks (s)")
    plt.ylabel("Occurrences")
    plt.show()


def get_maxima(df):
    peak_indices = argrelextrema(df[DIODE_VOL_COL_NAME].values, np.greater_equal, order = ORDER)[0]
    #peak_indices = find_peaks_cwt(df[DIODE_VOL_COL_NAME], np.arange(1, 10))
    maxima_rows = df.iloc[peak_indices]
    display_data(df, maxima_rows)
    return maxima_rows


def get_map_function(maxima_rows):
    maxima_rows[PREV_MAX_COL_NAME] = maxima_rows[DIODE_VOL_COL_NAME].shift()
    # Ignores first row
    maxima_rows = maxima_rows.iloc[1:, :]
    plt.scatter(maxima_rows[PREV_MAX_COL_NAME], maxima_rows[DIODE_VOL_COL_NAME], c='red', s=4)
    plt.suptitle("Measured Map Function")
    plt.xlabel("Voltage at Previous Peak (V)")
    plt.ylabel("Voltage at Peak (V)")
    plt.show()


if __name__ == "__main__":
    #df = pd.read_csv(FOLDER_ADDRESS + "/" + FILE_ADDRESS + ".csv")
    data = extract_raw_data("./data/7_1v30khz.csv")
    maxima = get_maxima(data)
    get_map_function(maxima)