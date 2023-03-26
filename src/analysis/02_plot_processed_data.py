#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#############################
# 02_plot_processed_data.py #
#############################

@authors: Verna Heikkinen, Aino Kuusi, Estanislao Porta

Plots the processed EEG data of the PSD intensity (averaged across all channels) vs frequency for each subject and for each group.

It is used for visual assessment of individual subjects and general group behaviour. Arguments used to run the script are added to pickle object.

Arguments
---------
    - eeg_tmp_data.pickle : pickle object 
        Object of pickle format containing the dataframe with the data as well as the metadata with the arguments used to run the 01_read_processed_data.py script.
    
    - control_plot_segment : int
        Define which of the segments from the task will be used for plotting.
    
    - drop_subs : bool
        Defines whether some subjects need to be excluded before plotting.
    
    - roi : str 
        Defines the Region Of Interest for more localized information (WIP - Not currently functional).
        
Returns
-------

    - eeg_tmp_data.pickle : pickle object 
        Object of pickle format containing the dataframe with the data as well as the metadata with the information about the arguments used to run this script.

# TODO: Remove hardcoded values of frequency and use from config_eeg
# TODO: violin plots?
# TODO: ROIs. Check this out for rois: https://www.nature.com/articles/s41598-021-02789-9
"""
import time
import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from config_common import figures_dir


def load_data():  
    with open("eeg_tmp_data.pickle", "rb") as fin:
        dataframe, metadata = pickle.load(fin)

    return dataframe, metadata

def define_freq_bands(metadata):
    if metadata["freq_band_type"] == 'thin':
        freqs = np.array([x for x in range(1, 39)])
    elif metadata["freq_band_type"] == 'wide':
        freqs = np.array([1, 3, 5.2, 7.6, 10.2, 13, 16, 19.2, 22.6, 26.2, 30, 34, 38.2]).T

    return freqs
 
def global_averaging(df, metadata, freqs):
#    subject_array_list = []
    global_averages = []
     
    # Transform data to array, change to logscale and re-shape to 2D. Calculate average across all channels per subject 
    for idx in df.index:
        subj_arr = np.array(df.loc[idx])[2:]
        subj_arr = 10*np.log10(subj_arr.astype(float))
        subj_arr = np.reshape(subj_arr, (64, freqs.size))
        #subj_arr  = subj_arr[:, 0:37]
        #TODO: check these channels
        if metadata["roi"] == 'Frontal': 
            subj_arr = subj_arr[0:22, :]
        GA = np.mean(subj_arr, axis=0)
        global_averages.append(GA)
        #TODO: same for ROIs?
        
    return global_averages
     
def create_df_for_plotting(df, metadata, freqs, global_averages):  

    #shoo=np.array(global_averages)
    plot_df = pd.DataFrame(np.array(global_averages), columns=freqs)
    plot_df = plot_df.set_index(df.index)
    plot_df.insert(0, "Subject", df['Subject'])
    plot_df.insert(1, "Group", df['Group'])
    
    # Slice the array based on the index of the segment to plot
    segment_index = metadata["control_plot_segment"]-1
    plot_df = plot_df[segment_index:len(df):metadata["segments"]] 
    
    # Drop subjects if needed
    # The following subjects have very low power in PASAT_1:
    subs_to_drop = ['15P', '19P', '31P', '36C', '08P', '31C']
    if metadata["drop_subs"]:
        for sub in subs_to_drop:
            plot_df = plot_df.drop(plot_df[plot_df['Subject']==sub].index)
   
    return plot_df 
     
def plot_control_figures(plot_df, metadata):
    '''
    Plot a figure with two subplots: one with individual patients and another with group means and SD
    '''  
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Define style
    plt.style.use('seaborn-darkgrid')  
    # Define title based on the arguments and metadata
    figure_title = f'Average PSD over all channels vs frequency. \nTask: {metadata["task"]}, Freq band: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]} \nUsing segment {metadata["control_plot_segment"]} out of {metadata["segments"]}, Region of interest: {metadata["roi"]}.'
    # Add figure title
    f.suptitle(figure_title)
    
    # Subplot 1
    for index,  row in plot_df.iterrows():
        if row['Group'] == 1:
            col = 'red'
        else:
            col = 'green'
        data = row[2:]
        data = np.array(data)
        ax1.plot(freqs, data.T, color=col, alpha=0.2)
        ax1.text(x=freqs[-1], y=data.T[-1], s=row['Subject'], horizontalalignment='left', size='small', color=col)
    
    ax1.set_ylabel('PSD (dB)')
    
    # Subplot 2
    #Calculate means of each group & plot
    group_means = plot_df.groupby('Group').mean(numeric_only=True)
    ax2.plot(freqs, group_means.iloc[0, :], 'g--', linewidth=1, label='Controls')
    ax2.plot(freqs, group_means.iloc[1, :], 'r-.', linewidth=1, label='Patients')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (dB)') #only if no channel scaling
    ax2.legend()
    
    # Calculate SD of each group & plot around means
    group_sd = plot_df.groupby('Group').std(numeric_only=True)
    c_plus = group_means.iloc[0, :] + group_sd.iloc[0, :]
    c_minus = group_means.iloc[0, :] - group_sd.iloc[0, :]
    ax2.fill_between(freqs, c_plus, c_minus, color='g', alpha=.2, linewidth=.5)
    
    p_plus = group_means.iloc[1, :] + group_sd.iloc[1, :]
    p_minus = group_means.iloc[1, :] - group_sd.iloc[1, :]
    ax2.fill_between(freqs, p_plus, p_minus, color='r', alpha=.2, linewidth=.5)

def save_fig(metadata):
    # Save fig to disk
    if metadata["normalization"]:
        fig_filename = f'psd-control-plot_{metadata["task"]}_{metadata["freq_band_type"]}_normalized.png'
    else:
        fig_filename = f'psd-control-plot_{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized.png'
    plt.savefig(os.path.join(figures_dir, fig_filename))
    metadata["psd-control-plot-filename"] = fig_filename
    print(f'\nINFO: Success! Figure "{fig_filename}" has been saved to folder {figures_dir}')
    return metadata

def export_data(dataframe, metadata):
    """
    Creates a pickle object containing the csv and the metadata so that other scripts using the CSV data can have the information on how was the data collected (e.g., input arguments or other variables).
    
    Input parameters
    ----------------
    - dataframe: pandas dataframe
            Each row contains the subject_and_task label, the group which it belongs to, and the PSD data (for the chosen frquency bands and for all channels) per subject_and_tasks
    - metadata: dictonary
                Contains the input arguments parsed when running the script     
    Output
    ------
    - eeg_tmp_data.pickle : pickle object
            pickle object which contains the dataframe and the metadata
    """
    with open("eeg_tmp_data.pickle", "wb") as f:
        pickle.dump((dataframe, metadata), f)
    print('INFO: Success! CSV data and metadata have been bundled into file "eeg_tmp_data.pickle".')
    
if __name__ == '__main__':
    
    # Save time of beginning of the execution to measure running time
    start_time = time.time()
    
    # Add arguments to be parsed from command line    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', type=bool, help="Define the verbosity of the output. Default is False", metavar='', default=False)
    
    roi_areas = ['All', 'Frontal', 'Occipital', 'FTC', 'Centro-parietal']
    parser.add_argument('--roi', type=str, choices=roi_areas, help="ROI areas to be plotted. Default value is 'All'.", metavar='', default='All')
    parser.add_argument('--drop_subs', type=bool, help='Drop some subjects from the plotting. Default is False', metavar='', default=False)
    parser.add_argument('--control_plot_segment', type=int, help='Define which number of segment to use: 1, 2, etc. Default is 1', metavar='', default=1)    
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()
    
    # Execute the submethods:
    # 1 - Read data
    dataframe, metadata = load_data()
    
    # 2 - Store arguments in dictionary object 'metadata'
    metadata["control_plot_segment"] = args.control_plot_segment
    if metadata["control_plot_segment"] > metadata["segments"]:
        raise IndexError(f'List index out of range. The segment you chose is not allowed for task {metadata["task"]}. Please choose a value between 1 and {metadata["segments"]}.')
    metadata["roi"] = args.roi
    metadata["drop_subs"] = args.drop_subs
    
    # 3 - Define Frequency bands
    freqs = define_freq_bands(metadata)
    
    # 4 - Do global averaging and ROI slicing
    global_averages = global_averaging(dataframe, metadata, freqs)

    # 5- Create DF for plotting
    plot_df = create_df_for_plotting(dataframe, metadata, freqs, global_averages)

    # 6 - Plot control plot
    plot_control_figures(plot_df, metadata)
    
    # 7 - Save active figure and add information to metadata
    metadata = save_fig(metadata)
    
    # 8 - Export pickle object
    export_data(dataframe, metadata)
    
    # Calculate time that the script takes to run
    execution_time = (time.time() - start_time)
    print('\n###################################################\n')
    print(f'Execution time of 02_plot_processed_data.py: {round(execution_time, 2)} seconds\n')
    print('###################################################\n')