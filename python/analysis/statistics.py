#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:18:46 2022

@author: aino

Plots histograms for each frequency band.
Performs the Wilcoxon signed rank test. 
"""
from readdata import dataframe as df
from plot import global_df as gdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# gdf = gdf.drop(['22P_PASAT_run1_2', '22P_PASAT_run1_1']) #We want matched controls and patients, 22P's control had bad data
gdf = gdf.drop(['22P_ec_1', '22P_ec_2', '22P_ec_3'])
patients = gdf.loc[gdf['Group']==1]
controls = gdf.loc[gdf['Group']==0]

plot = True


delta_p = patients.iloc[:, 1]
delta_c =controls.iloc[:, 1]
theta_p = patients.iloc[:, 2]
theta_c = controls.iloc[:, 2]
alpha_p = patients.iloc[:, 3]
alpha_c = controls.iloc[:, 3]
beta_p = patients.iloc[:, 4]
beta_c = controls.iloc[:, 4]
gamma_p = patients.iloc[:, 5]
gamma_c = controls.iloc[:, 5]
hgamma_p = patients.iloc[:, 6]
hgamma_c =controls.iloc[:, 6]


if plot:
    fig, axes = plt.subplots(2,6)
    sns.histplot(delta_c, kde=True, color='g', ax = axes[0][0], bins=15)
    sns.histplot(delta_p, kde=True, color='r', ax = axes[1][0], bins=15)
    sns.histplot(theta_c, kde=True, color='g', ax = axes[0][1], bins=15)
    sns.histplot(theta_p, kde=True, color='r', ax = axes[1][1], bins=15)
    sns.histplot(alpha_c, kde=True, color='g', ax = axes[0][2], bins=15)
    sns.histplot(alpha_p, kde=True, color='r', ax = axes[1][2], bins=15)
    sns.histplot(beta_c, kde=True, color='g', ax = axes[0][3], bins=15)
    sns.histplot(beta_p, kde=True, color='r', ax = axes[1][3], bins=15)
    sns.histplot(gamma_c, kde=True, color='g', ax = axes[0][4], bins=15)
    sns.histplot(gamma_p, kde=True, color='r', ax = axes[1][4], bins=15)
    sns.histplot(hgamma_c, kde=True, color='g', ax = axes[0][5], bins=15)
    sns.histplot(hgamma_p, kde=True, color='r', ax = axes[1][5], bins=15)
    
    # Set labels and titles
    for i in range(6):
        axes[1][i].set_ylabel('')
        axes[0][i].set_ylabel('')
        axes[0][i].set_xlabel('')
        axes[1][i].set_xlabel('')
    axes[0][0].set_ylabel('Controls')
    axes[1][0].set_ylabel('Patients')
    axes[0][0].title.set_text('Delta')
    axes[0][1].title.set_text('Theta')
    axes[0][2].title.set_text('Alpha')
    axes[0][3].title.set_text('Beta')
    axes[0][4].title.set_text('Gamma')
    axes[0][5].title.set_text('High gamma')
    
    
    fig.suptitle('Patients vs controls')
    fig.supxlabel('Power (dB)')
    fig.supylabel('Count')



res_delta = wilcoxon(x=delta_p, y=delta_c)
res_theta = wilcoxon(x=theta_p, y=theta_c)
res_alpha = wilcoxon(x=alpha_p, y=alpha_c)
res_beta = wilcoxon(x=beta_p, y=beta_c)
res_gamma = wilcoxon(x=gamma_p, y= gamma_c)
res_hgamma = wilcoxon(x=hgamma_p, y=hgamma_c)




