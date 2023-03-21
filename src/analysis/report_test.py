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
from config_common import figures_dir
# Create report
# Put header (static) - General metadata?

# Section 1: If control plot exists, add small desciption and control plot (specific metadata?)

# Section 2: plot of all the four ROC AUCs. It would be nice to plot the subplots separately eh? Put metadata specific for this? 
# Could add accuracies, TPR and something else here

def load_data():  
    # Read in dataframe and metadata
    with open("output.pickle", "rb") as fin:
        dataframe, metadata = pickle.load(fin)

    return dataframe, metadata

def create_report(metadata):
    # HTML report
    report = open('report.html', 'w')
    # General header
    report.write(f'''
    <!DOCTYPE html>
    <html>
    <head>
    	<title>mTBI-EEG - Analysis</title>
    </head>
    <body>
    	<h1>mTBI-EEG report</h1>
        <p>Metadata:</p>
    
    ''')  
    # Include the PSD Control Plots
    
    if "psd-control-plot" in metadata:
        control_plots = os.path.join(figures_dir, metadata["psd-control-plot"])
        print(control_plots)
        report.write(f'''
        <h2>PSD Averages - Control plot</h1>
        <p>Processed data is plotted in the figure below, to visually assess the data. The PSD for each frequency bin was averaged accross all the channels. The first subplot shows these averages per subject, enabling to identify outliers.</p>
        <p>In the second subplot, the PSD for each frequency bin was averaged accross all the channels and all the subjects within each group. The standard deviation for both groups is also displayed.</p>
        <img src="{control_plots}" class="center">
        ''')
    
    if "roc-plots" in metadata:
        roc_plots = os.path.join(figures_dir, metadata["roc-plots"])
        report.write(f'''   
        <h2>ROC Plots</h2>
        <p>Processed data was analyzed using four different ML classifiers. Validation was done using Stratified KFold Cross Validation. The subplots below show the ROC curves obtained using each of the classifiers.</p>
        <img src="{roc_plots}" class="center">
        ''')
                         
    report.write('<p> All metadata values shown below</p>')
    # Loop over the dictionary items and write each key-value pair in a separate row
    for key, value in metadata.items():
        report.write(f'<p>{key}: {value}</p>\n')
        
    report.write('''
        </body>
        </html>
        ''')
    report.close()
    print('Success! Report created')

if __name__ == "__main__":
    dataframe, metadata = load_data()
    create_report(metadata)