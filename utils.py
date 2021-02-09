# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:13:46 2021

@author: Fatemeh
"""

import pandas as pd
import numpy as np
import matplotlib


def remove_nans(list_1, list_2):
    """remove every corresponding element in list_2 that has nan value in list_1"""
    nans = np.isnan(list_1)
    nan_indecis = np.where(nans)[0]
    return np.delete(list_2, nan_indecis).tolist()


def date_to_num(X):
    """convert a list of datetime dates to a list of numbers with order"""
    Xnew = matplotlib.dates.date2num(X)
    return Xnew


def generate_datapoints(dates, mode, time_period, start_delay):
    """from the start date (+ start_delay), it calculates all the dates 'time_period' number of days apart from 
    each other, and converts it to a list of numbers if it is datetime"""
    dates = np.sort(dates)
    start = dates[0]
    end = dates[len(dates)-1]
    timepoints = []
    if (mode == 'date'):
        date = start + np.timedelta64(start_delay,'D')
    else:
        date = start + start_delay
    timepoints.append(start)
    while True:
        if (mode == 'date'):
            date =  date + np.timedelta64(time_period,'D')
        else:
            date = date + time_period
        timepoints.append(date)
        if(date >= end):
            break
        # print(date)
    try:
        timepoints = date_to_num(timepoints)
    except:
        pass
    return timepoints