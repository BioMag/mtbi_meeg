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
    # Create a for row df. Two patients and two controls. With random data 
    array_per_subject = np.random.rand(n_freqs * n_channels)
    
    df = pd.DataFrame({'Group': [1, 0, 1], 'Subjects': ["26P", "01C", "02P"], 'freq1_ch1': [1, 2, 3], 'freq1_ch2': [10, 11, 12], 'freq2_ch1': [1, 2, 3], 'freq2_ch2': [10, 11, 12]})
    
    metadata["roi"] = 'All'
    
    
    freqs = np.arange(0, 100, 1)

    expected_output = [np.mean(np.reshape(10 * np.log10(np.array(df.loc[idx])[2:].astype(float)), (2, freqs.size))) for idx in df.index]

    assert global_averaging(df, metadata, freqs) == expected_output

def test_global_averaging_with_empty_dataframe():
    df = pd.DataFrame()
    metadata = {'sample_rate': 1000, 'channels': 64}
    freqs = np.arange(0, 100, 1)

    assert global_averaging(df, metadata, freqs) == []

def test_global_averaging_with_nan_values():
    df = pd.DataFrame({'A': [np.nan, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan], 'C': [np.nan, np.nan, np.nan], 'D': [np.nan, np.nan, np.nan]})
    metadata = {'sample_rate': 1000, 'channels': 64}
    freqs = np.arange(0, 100, 1)

    assert global_averaging(df, metadata, freqs) == []

def test_global_averaging_with_different_frequencies():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
    metadata = {'sample_rate': 1000, 'channels': 64}
    freqs1 = np.arange(0, 100, 1)
    freqs2 = np.arange(0, 50, 1)

    expected_output1 = [np.mean(np.reshape(10 * np.log10(np.array(df.loc[idx])[2:].astype(float)), (64, freqs1.size))) for idx in df.index]
    expected_output2 = [np.mean(np.reshape(10 * np.log10(np.array(df.loc[idx])[2:].astype(float)), (64, freqs2.size))) for idx in df.index]

    assert global_averaging(df, metadata, freqs1) != global_averaging(df, metadata, freqs2)
    assert global_averaging(df, metadata, freqs2) == expected_output2

def test_create_df_for_plotting():
    pass

def test_plot_control_figures():
    pass

def test_save_fig():
    pass


