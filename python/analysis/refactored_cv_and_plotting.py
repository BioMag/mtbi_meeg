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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, RocCurveDisplay, auc
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC

from statistics import mean, stdev


# Pseudocode:
    # Read in dataframe and create X, y and groups
    # Split data according to the validation method
    # Output warning if only one class in fold (probability aaaround 26%)
        # If theres only one class in one fold, re-run the split?
    # Test with a new subjects.txt where classes are more unbalanced (e.g., 70-30)
    
    # Fit classifier
    # Cross Validate
    # Return validation results as outputs: true_positives, false_positives, accuracy
    # Would an aggregated confusion matrix from all the splits help out?
    # Plot ROC curves
    # If using the kernel, how should outputs be parsed?
    # Could I run one validation for all 4 models?    
    
#%% Arguments
verbosity = False
# Segments in the chosen task
segments = 3

# Define if we want to use CV with only one segment per subject (and no groups)
one_segment_per_subject = False

## Classifier
#classifier = LinearDiscriminantAnalysis(solver='svd')

# List of accuracy?
accuracies = []

# Interpoling True Positive Rate
interpole_tpr = []

# True Positive RateS
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
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

#%% Use all subjects and segments

# # Initialize Stratified Group K Fold
# sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True)
# sgkf_split = sgkf.split(X, y, groups) 

# Evaluate if there's any fold with only one class
# for i, (train_index, test_index) in enumerate(sgkf_split):
#     #print(f"Test:  index={test_index},\n{y[test_index]}")
#     values, counts = np.unique(y[test_index], return_counts=True)
#     if np.unique(y[test_index]).size == 1:
#         print(f"WARN: Fold {i} has only 1 class! ####")
#     elif verbosity == True: 
#         fold_size = y[test_index].size
#         if counts[0]<=counts[1]:
#             print(f"\nFold {i}:")
#             print(f'Class balance: {round(counts[0]/fold_size*100)}-{round(100-counts[0]/fold_size*100)}')
#         else:
#             print(f"\nFold {i}:")
#             print(f'Class balance: {round(counts[1]/fold_size*100)}-{round(100-counts[1]/fold_size*100)}')
# Note: There is a 27% chance that there's a fold with only one class. This can impact the classifier (especially LDA)
# >>> After 999 iterations, we found 267 folds with 1 class
         
#%% Use only one segment per subject

# Slice data
if one_segment_per_subject == True:
    X_one_segment = X[0:len(X):segments]
    y_one_segment = y[0:len(y):segments]
    groups_one_segment = groups[0:len(groups):segments]
    
    # Initialize Stratified K Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    skf_split = skf.split(X_one_segment, y_one_segment, groups_one_segment)
    # TODO: Double check if choosing another segment than the 1st breaks something? It shouldnt
    
    # Iterate over the splits and check the data
    for ii, (train_index, test_index) in enumerate(skf_split):
        #print(f"Test:  index={test_index},\n{y_one_segment[test_index]}")
        values, counts = np.unique(y_one_segment[test_index], return_counts=True)
        if np.unique(y_one_segment[test_index]).size == 1:
            print(f"WARN: Fold {ii} has only 1 class! ####")
        elif verbosity == True: 
            fold_size_one_segment = y_one_segment[test_index].size
            print(f"\nFold {ii}:")
            if counts[0]<=counts[1]:
                print(f'Class balance: {round(counts[0]/fold_size_one_segment*100)}-{round(100-counts[0]/fold_size_one_segment*100)}')
            else:
                print(f'Class balance: {round(counts[1]/fold_size_one_segment*100)}-{round(100-counts[1]/fold_size_one_segment*100)}')    

#%%


# Initialize Stratified Group K Fold
sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True)
sgkf_split = sgkf.split(X, y, groups) 

# Initialize figure for plottting
fig, ax = plt.subplots(figsize=(6, 6))

# Define classifier
classifier = SVC(probability=True)
for split, (train_index, test_index) in enumerate(sgkf_split):
    # Generate train and test sets for this split
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit classifier
    classifier.fit(X_train, y_train)
    
    # Create Receiver Operator Characteristics from the estimator for current split
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        drop_intermediate=True,
        name=f"ROC fold {split+1}",
        alpha=1, #transparency
        lw=1, #line width
        ax=ax
    )

    # Predict outcomes
    y_pred = classifier.predict(X_test).astype(int)
    # Estimate accuracy for this split (normalized), and append to list of accuracies
    accuracies.append(accuracy_score(y_test, y_pred))    
    
    # interpolate to build a series of 'y' values that correspond to linspace 'mean_fpr' 
    interpole_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    # Adds intercept just in case I guess
    interpole_tpr[0] = 0.0
    
    tprs.append(interpole_tpr) 
    aucs.append(viz.roc_auc)

    # Control if there's only one class in a fold
    values, counts = np.unique(y[test_index], return_counts=True)
    if np.unique(y[test_index]).size == 1:
        print(f"WARN: Fold {split} has only 1 class! ####")
    elif verbosity == True: 
        fold_size = y[test_index].size
        if counts[0]<=counts[1]:
            print(f"\nFold {split}:")
            print(f'Class balance: {round(counts[0]/fold_size*100)}-{round(100-counts[0]/fold_size*100)}')
        else:
            print(f"\nFold {split}:")
            print(f'Class balance: {round(counts[1]/fold_size*100)}-{round(100-counts[1]/fold_size*100)}')

# plt.scatter(viz.fpr, viz.tpr) shows how this is a step-wise function


# Calculate the mean 
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

average_accuracy = mean(accuracies)
accuracy_std = stdev(accuracies)
AUC = (round(mean(aucs),3), round(stdev(aucs), 3))
print(f"Average accuracy: {round(average_accuracy,3)}, Standard deviation of accuracies: {round(accuracy_std,3)}\nAUC = {AUC}")

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

# Plot chance curve
ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")

# Labels and axis
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Mean ROC curve with variability\n(Positive label 'Patients')",
)

# Force square ratio plot
ax.axis("square")
# Define legend location
ax.legend(loc="lower right")
plt.show()




# https://www.imranabdullah.com/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

#  What does roc_curve.from_estimator() object have
#from pprint import pprint
#pprint(vars(viz))
#{'ax_': <matplotlib.axes._subplots.AxesSubplot object at 0x7f15cc85b580>,
# 'estimator_name': 'ROC fold 10',
# 'figure_': <Figure size 432x432 with 1 Axes>,
# 'fpr': array([0.        , 0.11111111, 0.33333333, 0.33333333, 0.44444444,
#       0.44444444, 1.        , 1.        ]),
# 'line_': <matplotlib.lines.Line2D object at 0x7f15cc88b280>,
# 'pos_label': 1,
# 'roc_auc': 0.4814814814814815,
# 'tpr': array([0.        , 0.        , 0.        , 0.58333333, 0.58333333,
#       0.75      , 0.75      , 1.        ])}