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
        Object of pickle format containing the dataframe with the data
        and the metadata with the information about the arguments
        used to run this script.    
    - control_plot_segment : int
        Define which of the segments from the task will be used for plotting.  
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)
from config_common import figures_dir
from pickle_data_handler import PickleDataHandler
from config_eeg import channels, thin_bands, wide_bands


def initialize_argparser(metadata):
    """ Initialize argparser and add args to metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', choices=['True', 'False'], help='Define the verbosity of the output. Default: False', default=False)
    roi_areas = ['All', 'Frontal', 'Occipital', 'FTC', 'Centro-parietal']
    parser.add_argument('--roi', type=str, choices=roi_areas, help='ROI areas to be plotted. Default: All', default='All')
    parser.add_argument('--control_plot_segment', type=int, help='Define which number of segment to use: 1, 2, etc. Default: 1', metavar='', default=1)    
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()

    # Add the input arguments to the metadata dictionary        
    metadata["control_plot_segment"] = args.control_plot_segment
    if metadata["control_plot_segment"] > metadata["segments"]:
        raise IndexError(f'List index out of range. The segment you chose is not allowed for task {metadata["task"]}. Please choose a value between 1 and {metadata["segments"]}.')
    metadata["roi"] = args.roi
    return metadata
    
def define_freq_bands(metadata):
    if metadata["freq_band_type"] == 'thin':
        #freqs = np.array([x for x in range(1, 43)])
        freqs = np.array([bands[0] for bands in thin_bands])
    elif metadata["freq_band_type"] == 'wide':
        #freqs = np.array([1, 3, 5.2, 7.6, 10.2, 13, 16, 19.2, 22.6, 26.2, 30, 34, 38.2]).T
        freqs = np.array([bands[0] for bands in wide_bands])

    return freqs

def global_averaging(df, metadata, freqs):
    
    if df.isnull().values.any():
        raise ValueError("Error: There is at least one NaN value.") 
    
    global_averages = []
    
     # Transform data to array, change to logscale and re-shape to 2D. Calculate average across all channels per subject 
    for idx in df.index:
        subj_arr = np.array(df.loc[idx])[2:]
        subj_arr = 10 * np.log10(subj_arr.astype(float))
        if subj_arr.size == 0:
            raise ValueError("Error: Empty data array.")
        else: 
            try:
                subj_arr = np.reshape(subj_arr, (channels, freqs.size))
            except ValueError as e:
                print("Error: Data array has incorrect dimensions.")
                raise e
        if metadata["roi"] == 'Frontal': 
            subj_arr = subj_arr[0:22, :]
        GA = np.mean(subj_arr, axis=0)
        global_averages.append(GA)
        
    return global_averages
     
def create_df_for_plotting(df, metadata, freqs, global_averages):  

    plot_df = pd.DataFrame(np.array(global_averages), columns=freqs)
    plot_df = plot_df.set_index(df.index)
    plot_df.insert(0, "Subject", df['Subject'])
    plot_df.insert(1, "Group", df['Group'])
    
    # Slice the array based on the index of the segment to plot
    segment_index = metadata["control_plot_segment"] - 1
    plot_df = plot_df[segment_index:len(df):metadata["segments"]] 
    
    return plot_df 
     
def plot_control_figures(plot_df, metadata):
    '''
    Plot a figure with two subplots: one with individual patients and another with group means and SD
    '''  
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.style.use('seaborn-darkgrid')  
    figure_title = f'Average PSD over all channels vs frequency. \nTask: {metadata["task"]}, Freq band: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]} \nUsing segment {metadata["control_plot_segment"]} out of {metadata["segments"]}, Region of interest: {metadata["roi"]}.'
    f.suptitle(figure_title)
    
    # Subplot 1
    ax1.set_ylabel('PSD (dB)')
    for _, row in plot_df.iterrows():
        if row['Group'] == 1:
            col = 'red'
        else:
            col = 'green'
        data = row[2:]
        data = np.array(data)
        ax1.plot(freqs, data.T, color=col, alpha=0.2)
        ax1.text(x=freqs[-1], y=data[-1], s=row['Subject'], horizontalalignment='left', size='small', color=col)
        
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
    """ 
    Saves fig to disk
    """ 
    if metadata["normalization"]:
        fig_filename = f'psd-control-plot_{metadata["task"]}_{metadata["freq_band_type"]}_normalized.png'
    else:
        fig_filename = f'psd-control-plot_{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized.png'
    plt.savefig(os.path.join(figures_dir, fig_filename))
    metadata["psd-control-plot-filename"] = fig_filename
    print(f'\nINFO: Success! Figure "{fig_filename}" has been saved to folder {figures_dir}')
    return metadata


if __name__ == '__main__':
    
    # Save time of beginning of the execution to measure running time
    start_time = time.time()
    
    # Execute the submethods:
    # 1 - Read data
    handler = PickleDataHandler()
    dataframe, metadata = handler.load_data()
    
    # 2 - InitializeInitialize command line arguments and save arguments to metadata
    metadata = initialize_argparser(metadata)
    
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
    handler.export_data(dataframe, metadata)
    
    # Calculate time that the script takes to run
    execution_time = (time.time() - start_time)
    print('\n###################################################\n')
    print(f'Execution time of 02_plot_processed_data.py: {round(execution_time, 2)} seconds\n')
    print('###################################################\n')
