# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:16:14 2020

@author: Fatemeh
"""

import numpy as np, scipy.stats as st
import os, sys, glob
import pandas as pd
from fitness_coefficient_calcu import return_fitness_coef_freq_interpolated


def normal_dist_mean(mu, sigma, sample_num): 
    """input: mean and standard deviation and num of samples to draw.
    return the mean value of random numbers drown from normal distribution"""
    normal_samples = np.random.normal(mu, sigma, sample_num)
#    normal_samples = 1.0 * np.array(normal_samples)
    mean = np.mean(normal_samples)
    return mean 


def normal_dist_sampling_mean(df, sample_num):
    """for each entry (value) in df, the mean value of random values drown from
    normal distribution is replaced"""
    standard_div_rowwise = df.std(axis=1).to_numpy()
    df_to_array = df.to_numpy()
    print(df_to_array.shape)
    row = []
    for i in range(df_to_array.shape[0]):
        column = []
        for j in range(df_to_array.shape[1]):
            column.append(normal_dist_mean(df_to_array[i][j], standard_div_rowwise[i], sample_num))
        row.append(column)
    # pd.DataFrame(row, index = df.index).to_csv("N_distr_"+str(sample_num)+".csv")
    return row
    
    
def poisson(lambda_, sample_num):
    samples = np.random.poisson(lambda_, sample_num)
    return samples 


def Confidence_Interval(samples): 
    """returns 95% confidence interval for samples"""
    CIs = []
    for col in samples.columns:
        CIs.append(st.t.interval(0.95, len(samples[col].values) - 1, loc = np.mean(samples[col].values), scale = st.sem(samples[col].values)))
    return CIs
    

def poisson_dist_sampling_mean(df, sample_num):
    """for each entry (value) in df, the mean value of random values drown from
    poisson distribution is replaced"""
    df_to_array = df.to_numpy()
    print(df_to_array.shape)
    row = []
    for i in range(df_to_array.shape[0]):
        column = []
        for j in range(df_to_array.shape[1]):
            column.append(np.mean(poisson(df_to_array[i][j], sample_num)))
        row.append(column)
    row = pd.DataFrame(row, index = df.index)
    # row.to_csv("test/poisson_distr_"+str(sample_num)+".csv")
    return row


def remove_small_clusters(df, c):
    """remove the columns in which the summation of all the rows is less than c"""
    rowwise_sum = df.sum(axis = 0)
    for column in df.columns:
        if rowwise_sum[column] <= c:
            df.drop([column], axis=1, inplace=True)
    return df


def fitness_coefficient(file, iterations):
    """reads file and iterates fitness calculation with poisson error sampling, specified times
    and saves the confidence interval of fitness in a csv file"""
    df = pd.read_csv(file)
    df = df.set_index(df.columns[0])
    df = remove_small_clusters(df, 100)
    fitness_for_clusters = pd.DataFrame(columns=df.columns)
    for i in range(iterations):
        distributed_freq = poisson_dist_sampling_mean(df, 2000)
        # distributed_freq.to_csv(file.split(".csv")[0] + "_poisson.csv")
        resistance_coef, fitness_function = return_fitness_coef_freq_interpolated(distributed_freq, 1)
        for column in fitness_for_clusters.columns:
            fitness_for_clusters.at[i, column] = resistance_coef[list(fitness_for_clusters.columns).index(column)]
    CIs = Confidence_Interval(fitness_for_clusters)
    pd.DataFrame(CIs, index = df.columns).T.to_csv(str(file).split('.csv')[0] + '_CI_h_i.csv')


def read_from_path(path):
    for file in glob.glob(os.path.join(path, '*.csv')):
        print(file)
        fitness_coefficient(file)


if __name__ == '__main__':
    file = sys.argv[1]  
    fitness_coefficient(file, 100)
 

