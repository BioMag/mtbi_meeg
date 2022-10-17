#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:14:06 2022

@author: heikkiv

Does (spaghetti) plots of the log-PSDs. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from readdata import chosen_tasks, dataframe as df

#check https://www.python-graph-gallery.com/123-highlight-a-line-in-line-plot for deviations.


#TODO: vectorized data back to matrix (n*m), from which we should calculate global powers?
#So each df row now has [ch1_freq1, ..., ch64_freq89, ch2_freq1, ..., ch64_freq1, ...ch64_freq64] 

#-> revert these? or take from before? 

subject_array_list = [];
global_averages = [];

freqs=np.array([x for x in range(1,90)])


for idx in df.index:
    subj_data=df.loc[idx]
    subj_arr = np.array(subj_data)[2:len(subj_data)]
    subj_arr = 10*np.log10(subj_arr.astype(float))
    
    #reshape to 2D array again: (+ change to logscale)
    subj_arr = np.reshape(subj_arr, (64, 89)) #Rows=channels, cols=freqbands
    
    #calculate global average power accross all chs:
    GA = np.mean(subj_arr, axis=0)
    global_averages.append(GA)
    #TODO: same for ROIs?
    

#shoo=np.array(global_averages)
plot_df = pd.DataFrame(np.array(global_averages), columns=freqs)
plot_df.set_index(df.index)
plot_df['Group'] =  df['Group'].values
plot_df['Subject'] = df['Subject'].values

plot_df = plot_df.iloc[::2,:] #take every other value(=1obs. per subject)

# Change the style of plot
plt.style.use('seaborn-darkgrid')
plt.figure()

for subj in plot_df['Subject'].values: 
    data = plot_df.loc[lambda plot_df: plot_df['Subject']==subj]
    group = int(data['Group'])
    if group==1: 
        col='red' #change color based on clinical status
    else:
        col='green'
    
    i=list(plot_df['Subject'].values).index(subj)
    
    data=data.drop(['Group', 'Subject'], axis=1)
    plt.plot(freqs, data.values.T, color=col, alpha=0.4)
    plt.text(90,data.values.T[-1], subj, horizontalalignment='left', size='small', color=col)

#Calculate also means of controls and patients
group_means=plot_df.groupby('Group').mean()

plt.plot(freqs, group_means.iloc[0,:], marker='.', color='green', linewidth=1)
plt.plot(freqs, group_means.iloc[1,:], marker='.', color='red', linewidth=1)

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)') #only if no channel scaling
