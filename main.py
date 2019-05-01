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
POINTS_FOR_FIT = 10000

TIME = 'time'
GEN_V = 'gen_v'
DIODE_V = 'diode_v'
PREV_MAX = 'prev_max'

# # Maxima filtering parameters
# MAXIMA_TIMEDIFF_BINS = 20
# MAXIMA_TIMEDIFF_QUANTILE_BAR = 0.2


def extract_raw_data(filename):
    """
    Takes a data file and returns a useable data frame from it.
    :return: a data frame with columns: 'time', 'gen_v' which is the voltage on the generator
    and 'diode_v' which is the voltage on the diode.
    """
    raw_data = pd.read_csv(filename)
    data = pd.DataFrame({TIME: raw_data.iloc[:, 3],
                         GEN_V: raw_data.iloc[:, 4],
                         DIODE_V: raw_data.iloc[:, 10]})
    subsample_rate = np.ceil(data.shape[0] / MAX_ROWS)
    data = data.iloc[::subsample_rate, :]
    return data


def plot_raw_data(df):
    plt.plot(df[TIME], df[DIODE_V])
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
    """
    Displays data for manual verification of peak detection.
    :param data: the basic dataframe of voltage and
    :param maxima_rows:
    :return:
    """
    disp_df = data[(data[TIME] > 0) & (data[TIME] < 0.002)]
    disp_maxima = maxima_rows[(maxima_rows[TIME] > 0) & (maxima_rows[TIME] < 0.002)]
    plt.plot(disp_df[TIME], disp_df[DIODE_V], linewidth=0.5)
    plt.scatter(disp_maxima[TIME], disp_maxima[DIODE_V], s=4, c='red')
    plt.suptitle("Data Segment With Detected Maxima")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.show()
    timediffs = maxima_rows[TIME].diff().fillna(0)
    plt.hist(timediffs, rwidth=0.5)
    plt.suptitle("Histogram of Time Difference Between Maxima")
    plt.xlabel("Time Difference Between Consecutive Peaks (s)")
    plt.ylabel("Occurrences")
    plt.show()


def get_maxima(df):
    """
    Returns the data rows containing the detected maxima.
    :param df: a dataframe of time and voltages of generator and diode.
    :return: a similar dataframe for rows detected as peaks.
    """
    min_val = np.min(df[DIODE_V])
    # Makes diode_v relative to the minimum voltage measured in each measurement.
    df[DIODE_V] = df[DIODE_V] - min_val
    peak_indices = argrelextrema(df[DIODE_V].values, np.greater_equal, order=ORDER)[0]
    maxima_rows = df.iloc[peak_indices]
    return maxima_rows


def get_map_data(maxima_rows):
    """
    Returns a dataframe of maxima rows, with a column of previous maximum values.
    Each row is a datapoint for the map function.
    """
    maxima_rows[PREV_MAX] = maxima_rows[DIODE_V].shift()
    maxima_rows = maxima_rows[maxima_rows[DIODE_V] != maxima_rows[PREV_MAX]]
    # Ignores first row
    maxima_rows = maxima_rows.iloc[1:, :]
    return maxima_rows


def fit_map_function(map_data, range_min=-1.0, range_max=11.0):
    """
    Plots graphs relating to the map function.
    :param map_data: a dataframe of maximum rows with a column of previous maximum value
    :param range_min: the minimum of the range on which to plot the map function.
    :param range_max: the maximum of that range.
    (like output of get_map_data)
    """
    map_data = map_data[(map_data[PREV_MAX] >= range_min) & (map_data[PREV_MAX] <= range_max)]
    x = map_data[PREV_MAX]
    y = map_data[DIODE_V]
    coeffs = np.polyfit(x, y, deg=2)
    fit_x = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x)) / POINTS_FOR_FIT)
    fit_y = np.polyval(coeffs, fit_x)
    plt.scatter(x, y, c='red', s=4)
    plt.plot(fit_x, fit_y, linewidth=0.5)
    plt.suptitle("Measured Map Function")
    plt.xlabel("Voltage at Previous Peak (V)")
    plt.ylabel("Voltage at Peak (V)")


if __name__ == "__main__":
    #df = pd.read_csv(FOLDER_ADDRESS + "/" + FILE_ADDRESS + ".csv")
    data = extract_raw_data("./data/9_5v30khz.csv")
    maxima = get_maxima(data)
    map_data = get_map_data(maxima)
    fit_map_function(map_data, 0.5, 5.5)
    plt.show()