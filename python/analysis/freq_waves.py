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

theta = []
alpha = []
beta = []
gamma = []

# Get different waves from frequency bands
for i in data_matrices:
    theta_vectors = np.array(i[3:7])
    theta.append(np.sum(theta_vectors))
    alpha_vectors = np.array(i[7:11])
    alpha.append(np.sum(alpha_vectors))
    beta_vectors = np.array(i[11:34])
    beta.append(np.sum(beta_vectors))
    gamma_vectors = np.array(i[34:40])
    gamma.append(np.sum(gamma_vectors))

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

# Add labels
dataframes = [theta_dataframe, alpha_dataframe, beta_dataframe, gamma_dataframe]
for i in dataframes:
    i.insert(0, 'Group', groups)


# Theta/beta
theta_beta = theta_dataframe.div(beta_dataframe.values)


