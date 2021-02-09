# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:56:08 2020

@author: Fatemeh
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate
from utils import generate_datapoints, remove_nans


def spline_interpolate(x,y,xs):
    """x is datapoints, y is values and xs is the datapoints we need interpolation for"""
    cspl =  CubicSpline(x, y)
    cspl_dr = cspl.derivative() #pchip(x, y).derivative()
    return cspl_dr(xs), cspl(xs)


def calculate_cubic_derivatives(df, time_period):
    """caluclates derivative of cubic spline interpolation for each column over all weeks"""
    weeks_indices = generate_datapoints(df.index, 'index', time_period, 0)
    df = df.replace(0, np.nan)
    result = pd.DataFrame(index = weeks_indices)
    interp = pd.DataFrame(index = weeks_indices)
    for col in df.columns:
        no_nan_indices = remove_nans(df[col], df.index)
        no_nan_values = df[col].loc[no_nan_indices]
        try:
            result[col], interp[col] = spline_interpolate(no_nan_indices, no_nan_values, weeks_indices)
        except:
            continue
    return result.loc[df.index], interp.loc[df.index]


def calculate_cubic_derivatives_col(column, column_num, time_period):
    """caluclates derivative of cubic spline interpolation for each column over all weeks"""
    weeks_indices = generate_datapoints(column.index, 'index', time_period, 0)
    result = pd.DataFrame(index = weeks_indices)
    no_nan_indices = remove_nans(column, column.index)
    no_nan_values = remove_nans(column, column)
    result[column_num] = spline_interpolate(no_nan_indices, no_nan_values, weeks_indices)

    return result.loc[column.index]

    
    
def calculate_normal_freq(frequencies_df):
    """calculates the normalized frequencies for each row"""
    sum_df = frequencies_df.sum(axis=1)
    normal_freq = frequencies_df.div(sum_df.values, axis='index')
    return normal_freq


def integral(x, y):
    """return the integral for a set of points"""
    return integrate.cumtrapz(y, x, initial=0)


def calculate_resistance_coefficient(fitness_df):
    """calculates the average fitness over time for each column of input df"""
    columns = fitness_df.columns
    time = range(1, fitness_df.shape[0] + 1)
    coefficients = []
    for col in columns:
        coefficients.append(integral(time, fitness_df[col])[-1])
    c = fitness_df.shape[0]
    return [i / c for i in coefficients]


def calculate_resistance_coefficient_col(column):
    """calculates the average fitness for the input column"""
    no_nan_indices = remove_nans(column, column.index)
    # print(no_nan_indices)
    time = range(1, len(no_nan_indices) + 1)
    coefficients = integral(time, column.loc[no_nan_indices])[-1]
    return coefficients / len(no_nan_indices)


def return_resistance_coef_with_h(cluster_freq_df, time_period):
    """considers h, not h_i in g_i calculation"""
    fitness_function = pd.DataFrame()
    resistance_coefficient = []
    h = pd.DataFrame(cluster_freq_df.sum(axis = 1), index = cluster_freq_df.index)
    h_derivation, h_interp = calculate_cubic_derivatives(h, time_period)
    h_division = h_derivation.div(h)
    h_division.index = cluster_freq_df.index
    for col in cluster_freq_df.columns:
        cluster_freq = cluster_freq_df.loc[cluster_freq_df[col].ne(0).idxmax() + 1:]
        u = calculate_normal_freq(calculate_normal_freq(cluster_freq))
        u_derivation, u_intrp =  calculate_cubic_derivatives(u, time_period)
        print(u_derivation)
        u = u.replace(0, 0.01)
        h_ = h_division.loc[cluster_freq_df[col].ne(0).idxmax() + 1:][h_division.columns[0]]
        fitness_function[col] = (u_derivation[col] / u[col]) + h_
        resistance_coefficient.append(calculate_resistance_coefficient_col(fitness_function[col]))
    return resistance_coefficient, fitness_function


def replace_zeros_in_middle_with_nan(df):
    """replaces zeros after start and before end of a cluster with np.nan to make sure
    interpolation fill in for these values. The rest of zero values are replaced by 0.1.
    (some variants or clusters started later then others)"""
    for col in df.columns:
        list_ = [i for i, e in enumerate(list(df[col])) if e != 0] 
        if len(list_) > 0:
            for row, row_idx in enumerate(df.index):
                if (list_[0] < row < list_[-1]) and (df[col][row_idx] == 0):
                    df[col][row_idx] = np.nan
    df = df.replace(0, 0.1)
    return df


def return_fitness_coef_freq_interpolated(cluster_freq_df, time_period):
    """calculates fitness coefficient for each columns of df as a seperate cluster
    after spline interpolation"""
    fitness_function = pd.DataFrame(index = cluster_freq_df.index)
    resistance_coefficient = []
    cluster_freq_derivation, freq_interp = calculate_cubic_derivatives(replace_zeros_in_middle_with_nan(cluster_freq_df), time_period)
    freq_interp = freq_interp.where(freq_interp >= 0, 0.1)
    for col in cluster_freq_df.columns:
        start_index = cluster_freq_df[col].ne(0).idxmax() + 8
        cluster_freq_col = cluster_freq_df.loc[start_index:].replace(0, 0.1)
        freq_der = cluster_freq_derivation.loc[start_index:]
        freq_intr = freq_interp.loc[start_index:]
        u = calculate_normal_freq(calculate_normal_freq(cluster_freq_col))
        u_appr, u_interp =  calculate_cubic_derivatives(u, time_period)
        fitness_function[col] = u_appr.div(u_interp).add(freq_der.div(freq_intr))[col]
        resistance_coefficient.append(calculate_resistance_coefficient_col(fitness_function[col]))
    return resistance_coefficient, fitness_function


# if __name__ == '__main__':