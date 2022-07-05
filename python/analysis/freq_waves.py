#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:16:59 2022

@author: aino

Creates dataframes for different frequency bands (delta, theta, alpha, beta and gamma)
"""

from readdata import data_matrices, subjects_and_tasks, groups
import numpy as np
import pandas as pd

delta = []
theta = []
alpha = []
beta = []
gamma = []
all_vectors = [] # Contains n (n = subjects x chosen_tasks) vectors (lenght = 320)
all_ctp = [] # Same dimensions as all_vectors
all_gtp = []


# Get different waves from frequency bands
for i in data_matrices:
    # data_matrices is a list of 39 x 64 matrices(39 freq bands and 64 channels). len(data_matrices) = subjects x chosen_tasks
    all_bands = [] # 5 x 64 matrix (5 frequency bands and 64 channels) for each i
    
    # Calculate global total power and channel total power
    global_total_power = np.sum(i) # float
    channel_total_power = np.sum(i, axis=0) # Vector, lenght=64 (64 channels)
    
    delta_vectors = np.array(i[0:3]) # 3 x 64 matrix
    delta.append(np.sum(delta_vectors, axis = 0)) # Save delta bandpowers to delta matrix
    all_bands.append(np.sum(delta_vectors, axis=0)) # Save delta bandpowers to all_bands 
    
    theta_vectors = np.array(i[3:7])
    theta.append(np.sum(theta_vectors, axis = 0))
    all_bands.append(np.sum(theta_vectors, axis=0))
    
    alpha_vectors = np.array(i[7:11])
    alpha.append(np.sum(alpha_vectors, axis = 0))
    all_bands.append(np.sum(alpha_vectors, axis=0))
    
    beta_vectors = np.array(i[11:34])
    beta.append(np.sum(beta_vectors, axis = 0))
    all_bands.append(np.sum(beta_vectors, axis=0))
    
    gamma_vectors = np.array(i[34:40])
    gamma.append(np.sum(gamma_vectors, axis = 0))
    all_bands.append(np.sum(gamma_vectors, axis=0))
    
    all_vectors.append(np.concatenate(all_bands)) # Vectorize all_bands (size = 5 x 64 => length = 320) and add the vector to all_vectors
    
    # Divide bandpower for each channel by channel total power or global total power
    all_gtp.append(np.concatenate(np.divide(all_bands, global_total_power)))
    all_ctp.append(np.concatenate(np.divide(all_bands, channel_total_power))) 
    

# Create indices for dataframes 
indices = []
for i in subjects_and_tasks:
    i = i[0].rstrip()+'_'+i[1]
    indices.append(i)

# Create dataframes for each band and all bands
theta_dataframe = pd.DataFrame(np.array(theta), indices)
alpha_dataframe = pd.DataFrame(np.array(alpha), indices)
beta_dataframe = pd.DataFrame(np.array(beta), indices)
gamma_dataframe = pd.DataFrame(np.array(gamma), indices)
all_dataframe = pd.DataFrame(np.array(all_vectors), indices)
ctp_dataframe = pd.DataFrame(np.array(all_ctp), indices)
gtp_dataframe = pd.DataFrame(np.array(all_gtp), indices)

# Add labels
dataframes = [theta_dataframe, alpha_dataframe, beta_dataframe, gamma_dataframe, all_dataframe, ctp_dataframe, gtp_dataframe]
for i in dataframes:
    i.insert(0, 'Group', groups)





