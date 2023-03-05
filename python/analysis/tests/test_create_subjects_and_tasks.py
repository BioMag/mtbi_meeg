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

# Get the parent directory of the current file 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

module_name = "01_read_processed_data"
# Use importlib to import the module by name
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

if __name__ == '__main__':
    unittest.main()

