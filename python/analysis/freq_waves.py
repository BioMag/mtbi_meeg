#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:16:59 2022

@author: aino

Creates dataframes for different frequency bands (theta, alpha, beta and gamma)
"""

from readdata import data_matrices, subjects_and_tasks, groups
import numpy as np
import pandas as pd

delta = []
theta = []
alpha = []
beta = []
gamma = []
all_vectors = []

# Get different waves from frequency bands
for i in data_matrices:
    all_bands = []
    
    delta_vectors = np.array(i[0:3])
    delta.append(np.sum(delta_vectors, axis = 0))
    all_bands.append(np.sum(delta_vectors, axis=0))
    
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
    
    all_vectors.append(np.concatenate(all_bands))
    

# Create indices for dataframes 
indices = []
for i in subjects_and_tasks:
    i = i[0].rstrip()+'_'+i[1]
    indices.append(i)

# Create dataframes for each wave
theta_dataframe = pd.DataFrame(np.array(theta), indices)
alpha_dataframe = pd.DataFrame(np.array(alpha), indices)
beta_dataframe = pd.DataFrame(np.array(beta), indices)
gamma_dataframe = pd.DataFrame(np.array(gamma), indices)
all_dataframe = pd.DataFrame(np.array(all_vectors), indices)

# Add labels
dataframes = [theta_dataframe, alpha_dataframe, beta_dataframe, gamma_dataframe, all_dataframe]
for i in dataframes:
    i.insert(0, 'Group', groups)





