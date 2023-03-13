#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:58:55 2023

@author: portae1
"""

import unittest
import importlib
import os
import sys
import tempfile
import shutil
import numpy as np

# Get the parent directory of the current file 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the directory to the Python path
sys.path.append(parent_dir)

# Use importlib to import the module as a string to avoid conflicts with the numbers
module_name = "01_read_processed_data"
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

class TestReadProcessedData(unittest.TestCase):
    def test_read_processed_data(self):
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
        expected_shape = (len(subjects_and_tasks), 5696)
        assert np.shape(result) == expected_shape, f"Output has shape {np.shape(result)}, but expected shape is {expected_shape}"
    
        # Remove the temporary directory
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    unittest.main()

