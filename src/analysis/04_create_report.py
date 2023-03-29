#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:00:27 2023
Creates an HTML report with the images created in the previous step of the pipeline
@author: portae1
"""

import pickle
import os
import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from config_common import figures_dir, reports_dir
from pickle_data_handler import PickleDataHandler 
 
if not os.path.isdir(reports_dir):
    os.makedirs(reports_dir)

# Section 2: plot of all the four ROC AUCs. It would be nice to plot the subplots separately eh? Put metadata specific for this? 
# Could add accuracies, TPR and something else here

def create_report(metadata):
    # Define filename & open HTML file
    report_filename = f'report_{metadata["roc-plots-filename"][:-4]}.html'
    report_path = os.path.join(reports_dir, report_filename)
    report = open(report_path, 'w')
    
    # General header
    report.write(f'''
    <!DOCTYPE html>
    <html>
    <head>
    	<title>mTBI-EEG - Analysis</title>
    </head>
    <body>
    	<h1>mTBI-EEG report: {metadata["task"]} - {metadata["freq_band_type"]}</h1>
        <h2>Normalized - Not scaled</h2>
    ''')  
    # Include the PSD Control Plots  
    if "psd-control-plot-filename" in metadata:
        control_plots = os.path.join(figures_dir, metadata["psd-control-plot-filename"])
        report.write(f'''
        <h2>PSD Averages - Control plot</h1>
        <p>Processed data is plotted in the figure below, to visually assess the data. The PSD for each frequency bin was averaged accross all the channels. The first subplot shows these averages per subject, enabling to identify outliers.</p>
        <p>In the second subplot, the PSD for each frequency bin was averaged accross all the channels and all the subjects within each group. The standard deviation for both groups is also displayed.</p>
        <img src="{control_plots}" class="center">
        ''')
    # Include the ROC plots
    if "roc-plots-filename" in metadata:
        roc_plots = os.path.join(figures_dir, metadata["roc-plots-filename"])
        #TODO: check that file exists
        print(roc_plots)       
        report.write(f'''   
        <h2>ROC Plots</h2>
        <p>Processed data was analyzed using four different ML classifiers. Validation was done using Stratified KFold Cross Validation. The subplots below show the ROC curves obtained using each of the classifiers.</p>
        <img src="{roc_plots}" class="center">
        ''')
        print('I did something with the roc-plots')
    # Metadata section                     
    report.write('''
                 <h2>Metadata</h2>
                 <ul>
                 ''')
    # Loop over the dictionary items and write each key-value pair in a separate row
    for key, value in metadata.items():
        report.write(f'<li><b>{key}:</b> {value}</li>\n')
    # Close the unordered list and close the body section
    report.write('''
        </ul>
        </body>
        </html>
        ''')
    report.close()
    print(f'INFO: Success! File "{report_filename}" created')
    print('\n***\n')
if __name__ == "__main__":
    handler = PickleDataHandler()
    dataframe, metadata = handler.load_data()
    create_report(metadata)
    