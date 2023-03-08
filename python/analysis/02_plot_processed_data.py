#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:14:06 2022

@author: heikkiv

Does (spaghetti) plots of the log-PSDs. 

Note: it does not work for wide frequency bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Split in modules
# TODO: implement wide-frequency bands with boxplot?
# TODO: Implement logic for number of segments derived from using EC/EO or PASAT
# TODO: Remove hardcoded values of frequency
# TODO: violin plots?
# TODO: ROIs. Check this out for rois: https://www.nature.com/articles/s41598-021-02789-9

#%% 

# Initialize variables
subject_array_list = []
global_averages = []
freqs = np.array([x for x in range(1, 90)])

drop_subs = False
ROI = 'All' #One of 'All', 'Frontal', 'Occipital', 'FTC', 'Centro-parietal'

# Define the number of segments per task
number_of_segments = 3
# Define the number of segment which one wants to plot: 0 for first, 1 for second, 2 for third (if applicable)
segment_to_plot = 0

#%%

# Read in dataframe, using the column named 'index' as index column
df = pd.read_csv('dataframe.csv', index_col = 'Index')
#vectorized data back to matrix (n*m), from which we should calculate global powers
#So each df row now has [ch1_freq1, ..., ch64_freq89, ch2_freq1, ..., ch64_freq1, ...ch64_freq89] 

#%%

# Calculate averages across all channels per subject 
for idx in df.index:
    # Transform the data of each subject to np array 
    subj_arr = np.array(df.loc[idx])[2:]
    # Change to logscale
    subj_arr = 10*np.log10(subj_arr.astype(float))
    #reshape to 2D array again where rows=channels, cols=freqbands
    subj_arr = np.reshape(subj_arr, (64, 89))
    
    #TODO: check these channels
    if ROI == 'frontal': 
        subj_arr = subj_arr[0:22,:]
    # Calculate global average power accross all channels
    GA = np.mean(subj_arr, axis=0)
    global_averages.append(GA)
    #TODO: same for ROIs?
    
#%% 

#Create the DataFrame to be used for plotting
#shoo=np.array(global_averages)
plot_df = pd.DataFrame(np.array(global_averages), columns=freqs)
plot_df = plot_df.set_index(df.index)
plot_df.insert(0, "Subject", df['Subject'])
plot_df.insert(1, "Group", df['Group'])

# Slice the array based on which segment to plot
plot_df = plot_df[segment_to_plot:len(df):number_of_segments]

#%% 

# Drop subjects if needed
# The following subjects have very low power in PASAT_1:
subs_to_drop=['15P', '19P', '31P', '36C', '08P', '31C']
if drop_subs:
    for sub in subs_to_drop:
        plot_df = plot_df.drop(plot_df[plot_df['Subject']==sub].index)

#%% 

# Plot a figure with two subplots: one with individual patients and another with group means and SD
# Initialize figure and two subplots ax1 and ax2
f, (ax1, ax2) = plt.subplots(2, 1)
# Define style
plt.style.use('seaborn-darkgrid')
# Add title
f.suptitle(f'Average PSD over all channel vs frequency\nUsing segment {segment_to_plot+1} out of {number_of_segments}\nRegion of interest: {ROI}. Data is normalized')

# Subplot 1
# Iterate over rows in dataframe: define color based on group and plot in subplot (1, 1)
for index,  row in plot_df.iterrows():
    if row['Group'] == 1:
        col = 'red'
    else:
        col = 'green'
    data = row[2:]
    ax1.plot(freqs, data.T, color = col, alpha = 0.2)
    ax1.text(x = 90, y = data.values.T[-1], s = row['Subject'], horizontalalignment='left', size='small', color = col)

ax1.set_ylabel('PSD (dB)')

# Subplot 2
#Calculate means of each group 
group_means=plot_df.groupby('Group').mean()

# Plot means in subplot (2, 1)
ax2.plot(freqs, group_means.iloc[0,:], 'g--', linewidth=1, label='Controls')
ax2.plot(freqs, group_means.iloc[1,:], 'r-.', linewidth=1, label='Patients')

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('PSD (dB)') #only if no channel scaling
ax2.legend()

# Calculate SD of each group
group_sd=plot_df.groupby('Group').std()

# Add SD around the group means
c_plus=group_means.iloc[0,:]+group_sd.iloc[0,:]
c_minus=group_means.iloc[0,:]-group_sd.iloc[0,:]
ax2.fill_between(freqs, c_plus, c_minus, color='g', alpha=.2, linewidth=.5)

p_plus=group_means.iloc[1,:]+group_sd.iloc[1,:]
p_minus=group_means.iloc[1,:]-group_sd.iloc[1,:]
ax2.fill_between(freqs, p_plus, p_minus, color='r', alpha=.2, linewidth=.5)