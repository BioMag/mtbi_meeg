#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:37:19 2023

@author: aino

Runs scripts readdata.py, ROC_AUC.py and plot.py with chosen arguments.
"""

import subprocess

if __name__ == "__main__":
    # Choose task ('ec', 'eo', 'PASAT_1', 'PASAT_2')
    task = 'PASAT_2'
    # Choose frequency bands ('wide', 'thin')
    bands = 'thin'
    # Choose classifier ('LR', 'LDA', 'SVM')
    clf = 'LR'
    # Choose what to plot ('PSD', 'ROI')
    plots = 'PSD'
    
    roc = False
    plot = True

    #TODO: what else should be possible to choose? All subjects vs matched subjects?
    
    # Run readdata.py
    subprocess.call(f"python readdata.py --task {task} --freq_bands {bands}", shell=True)
    if roc:
        # Run ROC_AUC.py
        subprocess.call(f"python ROC_AUC.py --task {task} --clf {clf}", shell=True)
    
    if plot:
        subprocess.call(f"python plot.py --task {task} --freq_bands {bands} --plots {plots}", shell=True)
        
