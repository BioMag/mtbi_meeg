#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#################################
#    04_create_report.py        #
#################################

@author: Estanislao Porta
Creates an HTML report with the images created in the previous step of the pipeline

# TODO: I wonder if I should fetch the the data and create a report with the data and not with the png images.....?

# TODO: If control plots are not found, text should be included, not an error?
"""

import os
import sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)
from config_common import figures_dir, reports_dir
from pickle_data_handler import PickleDataHandler 
 
if not os.path.isdir(reports_dir):
    os.makedirs(reports_dir)


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
    ''')  
    if metadata["normalization"] and not metadata["scaling"]:
        report.write('<h2>Normalized - Notq scaled</h2>')
    if not metadata["normalization"] and metadata["scaling"]:
        report.write('<h2>Not normalized - Scaled</h2>')
    if not metadata["normalization"] and not metadata["scaling"]:
        report.write('<h2>Not normalized - Not scaled</h2>') 
    
    # Include the PSD Control Plots  
    if "psd-control-plot-filename" in metadata:
        control_plots = os.path.join(figures_dir, metadata["psd-control-plot-filename"])
        if not os.path.isfile(control_plots):            
            raise FileNotFoundError('Control plots were expected but are not found')
        
        report.write(f'''
        <h2>PSD Averages - Control plot</h1>
        
        <p>Processed data is plotted in the figure below, to visually assess the data. The PSD for each frequency bin was averaged accross all the channels. The first subplot shows these averages per subject, enabling to identify outliers.</p>
        
        <p>In the second subplot, the PSD for each frequency bin was averaged accross all the channels and all the subjects within each group. The standard deviation for both groups is also displayed.</p>
        <img src="{control_plots}" class="center">
        ''')
    else:
        print('INFO: No control plots')
    
    # Include the ROC plots
    if "roc-plots-filename" in metadata:
        roc_plots = os.path.join(figures_dir, metadata["roc-plots-filename"])
        
        if not os.path.isfile(roc_plots):            
            raise FileNotFoundError('ROC plots were expected but are not found')
        
        report.write(f'''   
        <h2>ROC Plots</h2>
        <p>Processed data was analyzed using four different ML classifiers. Validation was done using Stratified KFold Cross Validation. The subplots below show the ROC curves obtained using each of the classifiers.</p>
        <img src="{roc_plots}" class="center">
        ''')
    else:
        raise TypeError('No ROC plots')
    # Metrics section
    report.write('''
                 <h2>Metrics</h2>
                 ''')
    metrics = metadata["metrics"].drop('TPR', axis=1)
    report.write(metrics.to_html(index=False))
    # Metadata section                     
    report.write('''
                 <h2>Metadata</h2>
                 <ul>
                 ''')
    # Loop over the dictionary items and write each key-value pair in a separate row
    for key, value in metadata.items():
        if key == "metrics":
            continue
        report.write(f'<li><b>{key}:</b> {value}</li>\n')
    # Close the unordered list and close the body section
    report.write('''
        </ul>
        </body>
        </html>
        ''')
    report.close()
    print(f'INFO: File "{report_filename}" has been created in {reports_dir}.')


if __name__ == "__main__":
    handler = PickleDataHandler()
    dataframe, metadata = handler.load_data()
    create_report(metadata)
    