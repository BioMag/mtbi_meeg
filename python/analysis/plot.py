#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:28:20 2022

@author: aino

Plots ROI grand averages for a task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_ROI_grand_averages(df, bands, n_runs, n_channels):
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
    controls = len(global_df.loc[global_df['Group'] == 0])/n_runs
    patients = len(global_df.loc[global_df['Group']==1])/n_runs
    # Calculate and plot grand average patients vs controls 

    controls_total_power = np.sum(global_df.loc[global_df['Group']==0], axis = 0)
    controls_average = np.divide(controls_total_power[1:n_f_bands+1], controls)
    patients_total_power = np.sum(global_df.loc[global_df['Group']==1], axis = 0)    
    patients_average = np.divide(patients_total_power[1:n_f_bands+1], patients)



    axes[0].plot(bands, controls_average, label='Controls')
    axes[0].plot(bands, patients_average, label='Patients')
    axes[0].title.set_text('Global average')
    axes[0].legend()

    # Plot region of interest
    # Occipital lobe (channels 55-64)
    controls_sum_o = np.sum(occipital_df.loc[global_df['Group']==0], axis = 0)
    controls_average_o = np.divide(controls_sum_o[1:n_f_bands+1], controls)
    patients_sum_o = np.sum(occipital_df.loc[global_df['Group']==1], axis = 0)    
    patients_average_o = np.divide(patients_sum_o[1:n_f_bands+1], patients)

    axes[1].plot(bands, controls_average_o, label='Controls')
    axes[1].plot(bands, patients_average_o, label='Patients')
    axes[1].title.set_text('Frontal lobe')
    axes[1].legend()

    # Frontal lobe (channels 1-22 (?))
    controls_sum_f = np.sum(frontal_df.loc[global_df['Group']==0], axis = 0)
    controls_average_f = np.divide(controls_sum_f[1:n_f_bands+1], controls)
    patients_sum_f = np.sum(frontal_df.loc[global_df['Group']==1], axis = 0)    
    patients_average_f = np.divide(patients_sum_f[1:n_f_bands+1], patients)

    axes[2].plot(bands, controls_average_f, label='Controls')
    axes[2].plot(bands, patients_average_f, label='Patients')
    axes[2].title.set_text('Occipital lobe')
    axes[2].legend()

    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Normalized PSDs')
# # Plot band powers for a single channel and a single subject
# fig3, ax3 = plt.subplots()
# sub_df =log_df.loc[log_df.index==df.index[0]]
# sub_array = []# # Plot band powers for a single channel and a single subject
# fig3, ax3 = plt.subplots()
# sub_df =log_df.loc[log_df.index==df.index[0]]
# sub_array = []
# channel = 1 
# for i in range(n_f_bands):
#     sub_array.append(sub_df.iloc[:, channel-1+64*i])
# ax3.plot(channels, pd.DataFrame(sub_array))
# plt.title('Sub-'+df.index[0]+' Channel '+str(channel))
# channel = 1 
# for i in range(n_f_bands):
#     sub_array.append(sub_df.iloc[:, channel-1+64*i])
# ax3.plot(channels, pd.DataFrame(sub_array))
# plt.title('Sub-'+df.index[0]+' Channel '+str(channel))

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    df = pd.read_csv('/net/tera2/home/aino/work/mtbi-eeg/python/analysis/dataframe.csv')

    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    parser.add_argument('--ROI', type=bool, default=False)
    parser.add_argument('--freq_bands', type=str, help="wide, thin", default='wide')
    parser.add_argument('--task', type=str, help="ec, eo, PASAT_1 or PASAT_2")

    args = parser.parse_args()
    
    save_folder = "/net/tera2/home/aino/work/mtbi-eeg/python/figures"
    
    subjects = df.loc[:,'Subject']
    del df['Subject']
    # Number of runs
    if args.task == 'ec' or 'eo':
        n_runs = 3
    elif args.task == 'PASAT_1' or 'PASAT_2':
        n_runs = 2
    # Frequency band names
    if args.freq_bands == 'wide':
        n_f_bands = 6
        bands=['delta', 'theta', 'alpha', 'beta', 'gamma', 'high gamma']
    elif args.freq_bands == 'thin':
        n_f_bands = 89
        bands = [x for x in range(n_f_bands)]

    groups = df.loc[:, 'Group']
    n_channels = 64


        
    
    if args.ROI:
        fig, axes = plt.subplots(1,3)
        plot_ROI_grand_averages(df, bands, n_runs, n_channels)
        save_file = f"{save_folder}/ROI_{args.task}_{args.freq_bands}.pdf"
        plt.savefig(fname=save_file)
    #TODO: modify this so that the plots work if the frequency bands are changed
    #TODO: check https://www.python-graph-gallery.com/123-highlight-a-line-in-line-plot for deviations. Construct a dataframe? Move plotting to new script entirely?
    
    



