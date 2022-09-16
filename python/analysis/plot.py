#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:28:20 2022

@author: aino

Plots ROI grand averages for a task
THIS DOES NOT WORK YET
"""
from readdata import dataframe as df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


subjects = df.loc[:,'Subject']
tasks = []
# Get number of tasks/runs
for i in df.index:
    if subjects[0] in i:
        task = i.removeprefix(subjects[0]+'_')
        tasks.append(task)
groups = df.loc[:, 'Group']
n_f_bands = int(len(df.columns)/64)
n_channels = 64

if n_f_bands == 6:
    channels=['delta', 'theta', 'alpha', 'beta', 'gamma', 'high gamma']
else:
    channels = [x for x in range(n_f_bands)]
    
#TODO: modify this so that the plots work if the frequency bands are changed
#TODO: check https://www.python-graph-gallery.com/123-highlight-a-line-in-line-plot for deviations. Construct a dataframe? Move plotting to new script entirely?



log_df = np.log10(df.iloc[:, 2:n_f_bands*n_channels+1])
# ROI total powers for each frequency band
global_tot = []
frontal_tot = []
occipital_tot = []

# Get information for each channel
for i in range(n_channels):
    global_tot.append(log_df.iloc[:, 0+n_f_bands*i:n_f_bands+n_f_bands*i])
    if i < 23:
        frontal_tot.append(log_df.iloc[:, 0+n_f_bands*i:n_f_bands+n_f_bands*i])
    if i > 54:
        occipital_tot.append(log_df.iloc[:, 0+n_f_bands*i:n_f_bands+n_f_bands*i]) 
# global_tot is a list on dataframes (n_f_bands x (subjects + tasks)). Each element of this list represents a single channel
global_df = np.add(global_tot[0], global_tot[1])
frontal_df = np.add(frontal_tot[0], frontal_tot[1])
occipital_df = np.add(occipital_tot[0], occipital_tot[1])
# Sum the dataframes such that we get one dataframe (n_f_bands x (subjects + tasks)) and the total bandpower for each frequency band
for i in range(n_channels-3):
    global_df = np.add(global_df, global_tot[i+2])
    if i < 21:
        frontal_df = np.add(frontal_df, frontal_tot[i+2])
    if i < 6:
        occipital_df = np.add(occipital_df, occipital_tot[i+2])
# Problem: for channel 64 there are only 88 frequency bands?? 

#Divide the total bandpowers by the number of channels
global_df = np.divide(global_df, 63)
frontal_df = np.divide(frontal_df, 22)
occipital_df = np.divide(occipital_df, 9)
# Insert 'Group' column
global_df.insert(0, 'Group', groups)
frontal_df.insert(0, 'Group', groups)
occipital_df.insert(0, 'Group', groups)

# Calculate the number of patients and controls
controls = len(global_df.loc[global_df['Group'] == 0])/len(tasks)
patients = len(global_df.loc[global_df['Group']==1])/len(tasks)

#This is to be implemented

# # Plot band powers for a single channel and a single subject
# fig3, ax3 = plt.subplots()
# sub_df =log_df.loc[log_df.index==df.index[0]]
# sub_array = []
# channel = 1 
# for i in range(n_f_bands):
#     sub_array.append(sub_df.iloc[:, channel-1+64*i])
# ax3.plot(channels, pd.DataFrame(sub_array))
# plt.title('Sub-'+df.index[0]+' Channel '+str(channel))




# Calculate and plot grand average patients vs controls 

controls_total_power = np.sum(global_df.loc[global_df['Group']==0], axis = 0)
controls_average = np.divide(controls_total_power[1:n_f_bands+1], controls)
patients_total_power = np.sum(global_df.loc[global_df['Group']==1], axis = 0)    
patients_average = np.divide(patients_total_power[1:n_f_bands+1], patients)



fig, axes = plt.subplots(1,3)
axes[0].plot(channels, controls_average, label='Controls')
axes[0].plot(channels, patients_average, label='Patients')
axes[0].title.set_text('Global average')
axes[0].legend()

# Plot region of interest
# Occipital lobe (channels 55-64)
controls_sum_o = np.sum(occipital_df.loc[global_df['Group']==0], axis = 0)
controls_average_o = np.divide(controls_sum_o[1:n_f_bands+1], controls)
patients_sum_o = np.sum(occipital_df.loc[global_df['Group']==1], axis = 0)    
patients_average_o = np.divide(patients_sum_o[1:n_f_bands+1], patients)

axes[1].plot(channels, controls_average_o, label='Controls')
axes[1].plot(channels, patients_average_o, label='Patients')
axes[1].title.set_text('Frontal lobe')
axes[1].legend()

# Frontal lobe (channels 1-22 (?))
controls_sum_f = np.sum(frontal_df.loc[global_df['Group']==0], axis = 0)
controls_average_f = np.divide(controls_sum_f[1:n_f_bands+1], controls)
patients_sum_f = np.sum(frontal_df.loc[global_df['Group']==1], axis = 0)    
patients_average_f = np.divide(patients_sum_f[1:n_f_bands+1], patients)

axes[2].plot(channels, controls_average_f, label='Controls')
axes[2].plot(channels, patients_average_f, label='Patients')
axes[2].title.set_text('Occipital lobe')
axes[2].legend()


fig.supxlabel('Frequency (Hz)')
fig.supylabel('Normalized PSDs')