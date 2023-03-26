#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#############################
# test_01_read_processed_data.py #
#############################
@author: Estanislao Porta 

Tests the functions from module 01_read_processed_data.py

"""

import unittest
import importlib
import os
import sys
import tempfile
import shutil
import numpy as np

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)
from config_eeg import channels

analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))
sys.path.append(analysis_dir)
module_name = "01_read_processed_data" # Use importlib to import the module as a string to avoid conflicts with the numbers
read_processed_data = importlib.import_module(module_name)


class TestCreateSubjectsAndTasks(unittest.TestCase):
    def test_create_subjects_and_tasks(self):
        chosen_tasks = ['ec_1', 'ec_2', 'ec_3']
        subjects = ['01P', '02C']
        subjects_and_tasks = read_processed_data.create_subjects_and_tasks(chosen_tasks, subjects)

        # Check that the output is a list
        self.assertIsInstance(subjects_and_tasks, list)
    
        # Check that the output has the correct length
        self.assertEqual(len(subjects_and_tasks), len(chosen_tasks) * len(subjects))
    
        # Check that each element in the output is a tuple with two elements
        for element in subjects_and_tasks:
            self.assertIsInstance(element, tuple)
            self.assertEqual(len(element), 2)

class TestReadData(unittest.TestCase):
    def test_read_data(self):
        # Create temporary directory and dummy data
        tmp_dir = tempfile.mkdtemp()
        subjects_and_tasks = [('01P', 'ec_1'), ('01P', 'ec_2'), ('01P', 'ec_3')]
        freq_bands = 'thin'
        normalization = False
        
        # Create dummy data files in the temporary directory
        for subject, task in subjects_and_tasks:
            subject_dir = os.path.join(tmp_dir, f'sub-{subject}', 'ses-01', 'eeg', 'bandpowers')
            os.makedirs(subject_dir, exist_ok=True)
            # Create random data for 
            data = np.random.rand(89, 64)
            filename = f'{freq_bands}_{task}.csv'
            filepath = os.path.join(subject_dir, filename)
            np.savetxt(filepath, data, delimiter=',')
    
        # Call the method with the temporary directory and dummy data
        processed_data_dir = tmp_dir
        result = read_processed_data.read_data(subjects_and_tasks, freq_bands, normalization, processed_data_dir)
    
        # Check that the output has the expected shape
        expected_shape = (len(subjects_and_tasks), channels*38)
        assert np.shape(result) == expected_shape, f"Output has shape {np.shape(result)}, but expected shape is {expected_shape}"
    
        # Remove the temporary directory
        shutil.rmtree(tmp_dir)

class TestCreateDataFrame(self):
    create_data_frame

if __name__ == '__main__':
    unittest.main()

