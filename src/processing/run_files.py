#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:23:04 2023

Runs the scripts in the processing folder 

It could also be done in bash using something like 
    # Define the arguments for the first file
    arg1_vals=("eo" "ec" "PASAT_1" "PASAT_2")
    arg2_vals=("thin" "wide")
    
    # Call the first Python file with each set of arguments
    for (( i=0; i<${#arg1_vals[@]}; i++ )); do
        python file1.py --task "${arg1_vals[i]}" --freq_band_type "${arg2_vals[i]}"
        # Call the second Python file without any arguments
        python file2.py
    done
@author: portae1
"""

import subprocess
import time
import re

# Flag used for test run
TEST_RUN = True


# Save time of beginning of the execution to measure running time
here_start_time = time.time()

# Define a list of tuples containing the different argument combinations to use
#subjects = ['10C', '30P']
arg_set = [('--freq_band_type', 'thin'),
            ('--freq_band_type', 'wide'),]

subject_pattern = r'^\d{2}[PC]'   
try:
    with open('subjects.txt', 'r') as subjects_file:
        subjects = [line.rstrip() for line in subjects_file.readlines()]
        # Assert that each line has the expected format
        for line in subjects:
            assert re.match(subject_pattern, line), f"Subject '{line}' does not have the expected format."

except FileNotFoundError as e:
    print("The file 'subjects.txt' does not exist in the current directory. The program will exit.")
    raise e

if TEST_RUN:
    subjects = ['10C', '11P']

for subject in subjects:
    print(f'### \nRunning using subject {subject}...\n')
    # Call the first Python 
    subprocess.run(['python3', '01_freqfilt.py', subject])
    print('Finished 01\n')
    # Call the second Python file
    subprocess.run(['python3', '02_ica.py', subject])
    print('Finished 02\n')
    # Call the third script
    subprocess.run(['python3', '03_psds.py', subject])
    print('Finished 03\n')
    # Create bandpowers
    for arg in arg_set:
        subprocess.run(['python3', '04_bandpower.py', subject] + list(arg))


# Calculate time that the script takes to run
here_execution_time = (time.time() - here_start_time)
print('\n###################################################')
print(f'Execution time of everything is: {round(here_execution_time/60,1)} minutes')
print(f'Average time is {round(here_execution_time/len(subjects),1)} seconds per subject')
print('###################################################\n')
