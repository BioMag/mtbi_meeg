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
#from readdata import chosen_tasks, dataframe as df

#vectorized data back to matrix (n*m), from which we should calculate global powers
#So each df row now has [ch1_freq1, ..., ch64_freq89, ch2_freq1, ..., ch64_freq1, ...ch64_freq64] 

#-> revert these
df = pd.read_csv('dataframe.csv')
segments = 3
df = df[2:len(df):segments]


subject_array_list = [];
global_averages = [];

drop_subs=False
ROI = 'All' #One of 'All', 'Frontal', 'Occipital', 'FTC', 'Centro-parietal'

freqs=np.array([x for x in range(1,90)]) #todo: pls no hardcoded values!

#check this out for rois: https://www.nature.com/articles/s41598-021-02789-9

for idx in df.index:
    subj_data=df.loc[idx]
    subj_arr = np.array(subj_data)[3:len(subj_data)]
    subj_arr = 10*np.log10(subj_arr.astype(float))
    
    #reshape to 2D array again: (+ change to logscale)
    subj_arr = np.reshape(subj_arr, (64, 89)) #Rows=channels, cols=freqbands
    
    if ROI == 'frontal': #TODO: check these channels
        subj_arr = subj_arr[0:22,:]
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

#The following subjects have very low power in PASAT_1:
subs_to_drop=['15P', '19P', '31P', '36C', '08P', '31C']
if drop_subs:
    for sub in subs_to_drop:
        plot_df = plot_df.drop(plot_df[plot_df['Subject']==sub].index)

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
    plt.plot(freqs, data.values.T, color=col, alpha=0.2)
    plt.text(90,data.values.T[-1], subj, horizontalalignment='left', size='small', color=col)

#Calculate also means of controls and patients
group_means=plot_df.groupby('Group').mean()

plt.plot(freqs, group_means.iloc[0,:], 'g--', linewidth=1, label='Controls')
plt.plot(freqs, group_means.iloc[1,:], 'r-.', linewidth=1, label='Patients')

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)') #only if no channel scaling
plt.title(f'{ROI}')
plt.legend()

#Make SD plot
plt.figure()
group_sd=plot_df.groupby('Group').std()


plt.plot(freqs, group_means.iloc[0,:], 'g--', linewidth=1, label='Controls')
plt.plot(freqs, group_means.iloc[1,:], 'r-.', linewidth=1, label='Patients')

c_plus=group_means.iloc[0,:]+group_sd.iloc[0,:]
c_minus=group_means.iloc[0,:]-group_sd.iloc[0,:]
plt.fill_between(freqs, c_plus, c_minus, color='g', alpha=.2, linewidth=.5)

p_plus=group_means.iloc[1,:]+group_sd.iloc[1,:]
p_minus=group_means.iloc[1,:]-group_sd.iloc[1,:]
plt.fill_between(freqs, p_plus, p_minus, color='r', alpha=.2, linewidth=.5)

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)') #only if no channel scaling
plt.legend()


#TODO: violin plots?






