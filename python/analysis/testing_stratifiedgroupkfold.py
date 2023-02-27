#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:58:23 2023

@author: portae1
"""
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold

# Pseudocode:
    # Read in dataframe and create X, y and groups
    # Split data according to the validation method
    # Visualize folds
    # If theres only one class in one fold, re-run the split?
    # Test with a new subjects.txt where classes are more unbalanced (e.g., 70-30)
    
    # Fit classifier
    # Cross Validate 
    # Return validation results as outputs: true_positives, false_positives, accuracy
    # Plot ROC curves
#%% 

# Read in dataframe and create X, y and groups
dataframe = pd.read_csv('dataframe.csv', index_col = 'Index')
## Define features, classes and groups
X = dataframe.iloc[:,2:]
y = dataframe.loc[:, 'Group']
groups = dataframe.loc[:, 'Subject']

folds = 10

#%% 
## (Optional)
    # Define what strategy to use based on the subjects balance

#%% Split data
# 1 - Split data using Stratified and Group approach
skf = StratifiedGroupKFold(n_splits=folds, shuffle=True)
split = skf.split(X, y, groups) 

for i, (train_index, test_index) in enumerate(split):
    print(f"\nFold {i}:")
    print(f"Test:  index={test_index},\n{y[test_index]}")
    values, counts = np.unique(y[test_index], return_counts=True)
    if np.unique(y[test_index]).size == 1:
        print(f"WARN: Fold {i} has only 1 class! ####")
    else: 
        
        if counts[0]<=counts[1]:
            print(f'Class balance: {round(counts[0]/(counts[0]+counts[1])*100)}-{round(100-counts[0]/(counts[0]+counts[1])*100)}')
        else:
            print(f'Class balance: {round(counts[1]/(counts[0]+counts[1])*100)}-{round(100-counts[1]/(counts[0]+counts[1])*100)}')
    
    #print(f"Test:  index={test_index},\n{y[test_index]}")
    #print(f"  Train: index={train_index}, group={group[train_index]}")
    
## If stratified is false (should give pretty much the same):
    # something something StGroupKFold()    
## To use only one subject per group:
    # redefine X and y
    # something something StratifiedKFold()
    # Double check if choosing another segment than the 1st breaks something? It shouldnt
# 

#%%
 
   