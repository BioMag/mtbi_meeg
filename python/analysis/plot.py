#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:28:20 2022

@author: aino

Plots ROI grand averages for a task

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
n_f_bands = 6
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
for i in range(n_f_bands):
    global_tot.append(np.sum(log_df.iloc[:, 0+64*i:64+64*i], axis=1))
    frontal_tot.append(np.sum(log_df.iloc[:, 0+64*i:22+64*i], axis=1))
    occipital_tot.append(np.sum(log_df.iloc[:,60+64*i:63+64*i], axis=1))
# ROI averages
global_df = np.divide(np.transpose(pd.DataFrame(global_tot)), 64)
frontal_df = np.divide(np.transpose(pd.DataFrame(frontal_tot)), 22)
occipital_df = np.divide(np.transpose(pd.DataFrame(occipital_tot)), 3)
# Insert 'Group' column
global_df.insert(0, 'Group', groups)
frontal_df.insert(0, 'Group', groups)
occipital_df.insert(0, 'Group', groups)

controls = len(global_df.loc[global_df['Group'] == 0])/len(tasks)
patients = len(global_df.loc[global_df['Group']==1])/len(tasks)


# Plot band powers for a single channel and a single subject
fig3, ax3 = plt.subplots()
sub_df =log_df.loc[log_df.index==df.index[0]]
sub_array = []
channel = 1 
for i in range(n_f_bands):
    sub_array.append(sub_df.iloc[:, channel-1+64*i])
ax3.plot(channels, pd.DataFrame(sub_array))
plt.title('Sub-'+df.index[0]+' Channel '+str(channel))




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