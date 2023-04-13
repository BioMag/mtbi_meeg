#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:23:04 2023

It runs the scripts:
    01_read_processed_data.py,
    02_plot_processed_data,
    03_fit_classifier_and_plot.py,
    04_create_report

It will:
- Read processed data for all subjects in subjects.txt
- Plot control plots and save them to the 'figures_dir'
- Fit classifiers, plot ROCs, and save the ROC plots and the metrics to a pickle object
- Create an html report in the 'reports_dir'

"""

import subprocess

# Define a list of tuples containing the different argument combinations to use
arg_sets = [
#('--task', 'eo', '--freq_band_type', 'thin'),
#            ('--task', 'ec', '--freq_band_type', 'thin'),
#            ('--task', 'PASAT_1', '--freq_band_type', 'thin'),
#            ('--task', 'PASAT_2', '--freq_band_type', 'thin'),
            ('--task', 'eo', '--freq_band_type', 'wide'),
            ('--task', 'ec', '--freq_band_type', 'wide'),
            ('--task', 'PASAT_1', '--freq_band_type', 'wide'),
            ('--task', 'PASAT_2', '--freq_band_type', 'wide'),]

for arg_set in arg_sets:
    print(f'### \nRunning using {arg_set[1]} and {arg_set[3]}...')
    # Call the first Python file with each set of arguments
    subprocess.run(['python3', '01_read_processed_data.py'] + list(arg_set))
    # Call the second Python file without any arguments
    subprocess.run(['python3', '02_plot_processed_data.py'])
    # Call the third script, no arguments
    subprocess.run(['python3', '03_fit_classifier_and_plot.py'])
    # Create report, no arguments
    subprocess.run(['python3', '04_create_report.py'])

print('Finished  running all tasks and freq_band, normalized, not scaled')
