#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#############################
# test_02_plot_processed_data.py #
#############################
@author: Estanislao Porta 

Tests the functions from module 02_plot_processed_data.py
Use `python3 -m pytest test_02_plot_processed_data.py` to run it from terminal
"""


import pytest
import importlib
import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)
from config_eeg import channels

analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))
sys.path.append(analysis_dir)
module_name = "02_plot_processed_data" # Use importlib to import the module as a string to avoid conflicts with the numbers
plot_processed_data = importlib.import_module(module_name)

def test_load_pickle_data():
    #possibly moved to a class
    pass

def test_define_freq_bands():
    # I dont think there is anything really to be tested here. It should actually be moved to the config-eeg
    pass


def test_global_averaging_with_sample_data():
    # TODO: Test for wide freqs!
    freqs = np.array([x for x in range(1, 39)])   

    array_subjects = np.random.rand(3, len(freqs) * channels)
    df = pd.DataFrame({'Group': [1, 0, 1], 'Subjects': ["26P", "01C", "02P"]})
    df = pd.concat([df, pd.DataFrame(array_subjects)], axis=1)
    metadata = {"roi": 'All'}
    expected_output = []
    
    for idx in df.index:
        subj_arr = np.array(df.loc[idx])[2:]
        subj_arr = 10*np.log10(subj_arr.astype(float))
        subj_arr = np.reshape(subj_arr, (channels, freqs.size))
        
        if metadata["roi"] == 'Frontal': 
            subj_arr = subj_arr[0:22, :]
        GA = np.mean(subj_arr, axis=0)
        expected_output.append(GA)    

    actual_output = plot_processed_data.global_averaging(df, metadata, freqs)
    assert len(expected_output) == len(actual_output)
    assert all(tuple(a) == tuple(b) for a, b in zip(actual_output, expected_output)), "The actual output does not match the expected output."
    
def test_global_averaging_with_empty_dataframe():
    # The pickle data handler should already be considering these issues
#    metadata = {"roi": 'All'}
#    freqs = np.array([x for x in range(1, 39)])   
#    array_subjects = np.empty(3, len(freqs) * channels)
#    df = pd.DataFrame({'Group': [1, 0, 1], 'Subjects': ["26P", "01C", "02P"]})
#    df = pd.concat([df, pd.DataFrame(array_subjects)], axis=1)
#    assert plot_processed_data.global_averaging(df, metadata, freqs) == []
    pass
def test_global_averaging_with_nan_values():
#    df = pd.DataFrame({'A': [np.nan, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan], 'C': [np.nan, np.nan, np.nan], 'D': [np.nan, np.nan, np.nan]})
#    metadata = {'sample_rate': 1000, 'channels': 64}
#    freqs = np.array([x for x in range(1, 39)])   
#
#    assert plot_processed_data.global_averaging(df, metadata, freqs) == []
    pass
def test_global_averaging_with_different_frequencies():
    # Try with wide freqs
    pass

def test_create_df_for_plotting():
    pass

def test_plot_control_figures():
    pass

def test_save_fig():
    pass


