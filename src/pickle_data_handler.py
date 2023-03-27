#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:09:59 2023

@author: portae1
"""

import pickle
import pandas as pd

class PickleDataHandler:
    
    
    @staticmethod
    def export_data(dataframe, metadata):
        if dataframe.empty:
            raise ValueError("The dataframe cannot be empty.")
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")
        if not metadata:
            raise ValueError("The metadata file cannot be empty.")
            
        try: 
            with open("eeg_tmp_data.pickle", "wb") as f:
                pickle.dump((dataframe, metadata), f)
        except (TypeError, IOError) as e:
            print(f'An error occurred: {e}')
            return False
        
        print('INFO: Success! CSV data and metadata have been bundled into file "eeg_tmp_data.pickle".')
        return True
    
    @staticmethod
    def load_data():  
        try:
            with open("eeg_tmp_data.pickle", "rb") as fin:
                dataframe, metadata = pickle.load(fin)
        except FileNotFoundError:
            print("The file 'eeg_tmp_data.pickle' does not exist in the current directory. The program will exit.")
            return False
        except (IOError, TypeError) as e:
            print(f'An error occurred: {e}')
            return False
        
        if dataframe.empty:
            raise ValueError("The dataframe is empty.")
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")
        if not metadata:
            raise ValueError("The metadata file cannot be empty.")
            
        print('INFO: Success! CSV data and metadata have been read in from file "eeg_tmp_data.pickle".')
        return dataframe, metadata