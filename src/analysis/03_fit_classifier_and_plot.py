#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#################################
# 03_fit_classifier_and_plot.py #
#################################

@authors: Verna Heikkinen, Aino Kuusi, Estanislao Porta

Takes the processed data, splits it in folds according to KFold cross validation and fits a classifier.
It evaluates the performance of the fit by plotting each classifiers mean ROC curve. 

Arguments
---------
    - output.pkl : pickle object
        Object of pickle format containing the dataframe with the data and the metadata with the information about the arguments used to run the 01_read_processed_data script.    
        
    - seed : int
        
    - folds : str
        
    - scaling : bool
    - scaling_method : str
    - one_segment_per_task : bool
        whether one or all segments of the task will be used
    - which_segment : int
        If one_segment_per_task is True, this defines which of the segments will be used
        
Returns
-------

    - figure  : pickle object 
        Object of pickle format containing the dataframe with the data and the metadata with the information about the arguments used to run this script.
    - metadata?



# TODO: Return validation results as outputs: true_positives, false_positives, accuracy
# TODO: Embed metadata to image and/or as output file
# TODO: Create a report?
# TODO: Add logging?
"""
import sys
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import csv
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, RocCurveDisplay, auc
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from statistics import mean, stdev
from datetime import datetime

processing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(processing_dir)
from config_common import reports_dir, figures_dir
if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir)

#%%
    
def load_data():
    """
    Load pickle data and initialize variables 
    return dataframe, metadata
    """
    # Read in dataframe and metadata
    with open("output.pickle", "rb") as fin:
        dataframe, metadata = pickle.load(fin)
        
    return dataframe, metadata
    
def initialize_variables(metadata):
    """
    Initializes variables 
    return metadata
    """

    # Random seed for the classifier
    # Note: different sklearn versions could yield different results
    seed = 8
    metadata["seed"] = seed
       
    # Number of folds to be used during Cross Validation
    folds = 10
    metadata["folds"] = folds
    
    # Segments in the chosen task
    if (metadata["task"] in ('eo', 'ec')):
        segments = 3
    elif (metadata["task"] in ('PASAT_1', 'PASAT_2')):
        segments = 2
    metadata["segments"] = segments
    
    # Define if we want to use CV with only one segment per subject (and no groups)
    one_segment_per_task = False
    metadata["one_segment_per_task"] = one_segment_per_task
    
    # Which segment to be used when using only one segment for fitting
    which_segment = 0
    metadata["which_segment"] = which_segment
    
    # Standardize data
    scaling = False
    metadata["scaling"] = scaling
    scaling_method = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    metadata["scaling_method"] = scaling_method[2]
    return metadata

def initialize_cv(dataframe, metadata):
    """
    Initialize CrossValidation and get data splits as a list
    return X, y, groups, metadata, data_split
    """
    ## Define features, classes and groups
    X = dataframe.iloc[:,2:]
    y = dataframe.loc[:, 'Group']
    groups = dataframe.loc[:, 'Subject']
    
    # Slice data
    if metadata["one_segment_per_task"] == True:
        # Removes (segments-1) rows out of the dataframe 
        X = X[metadata["which_segment"]:len(X):metadata["segments"]]
        y = y[metadata["which_segment"]:len(y):metadata["segments"]]
        groups = groups[metadata["which_segment"]:len(groups):metadata["segments"]]
        
        # Initialize Stratified K Fold
        skf = StratifiedKFold(n_splits=metadata["folds"], shuffle=True, random_state= metadata["seed"])
        data_split = list(skf.split(X, y, groups))
    else:
        # Initialize Stratified Group K Fold
        sgkf = StratifiedGroupKFold(n_splits=metadata["folds"], shuffle=True, random_state=metadata["seed"])
        data_split = list(sgkf.split(X, y, groups))

    return X, y, groups, data_split

def fit_and_plot(X, y, groups, classifiers, data_split, metadata):
    # I could get classifiers from config_eeg and avoid defining that here
    """
    Loops over all classifiers and accordinat to the CV.
    Plots the results in subplots, saves figure
    return metadata
    """
    # Initialize figure for plottting
    fig, axs = plt.subplots(nrows=2, ncols=2, 
                            sharex = True, sharey = True, 
                            figsize=(10,10))
    tpr_per_classifier = []
    ii=0
    
    ## List containing the accuracy of the fit for each split - WIP
    #accuracies = []
    #
    ## Interpoling True Positive Rate
    #interpole_tpr = []
    
    for ax, (name, clf) in zip(axs.flat, classifiers):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Fit and do CV
        for split, (train_index, test_index) in enumerate(data_split):
            # Generate train and test sets for this split
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            if metadata["scaling"] and not metadata["normalization"]:
                scaler = metadata["scaling_method"]
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif metadata["scaling"] and metadata["normalization"]:
                raise TypeError("You are trying to scale data that has been already normalized.")
            #print(f'Split {split+1}\nShape of X_train is {X_train.shape[0]} x {X_train.shape[1]}')
            #print(f'Shape of X_test is {X_test.shape[0]} x {X_test.shape[1]}')
        
            # Control if there's only one class in a fold
            values, counts = np.unique(y[test_index], return_counts=True)
            if np.unique(y[test_index]).size == 1:
                print(f"WARN: Split {split+1} has only 1 class in the test set, skipping it. ####")
                continue
            elif verbosity == True: 
                fold_size = y[test_index].size
                if counts[0]<=counts[1]:
                    print(f"\nFold {split}:")
                    print(f'Class balance: {round(counts[0]/fold_size*100)}-{round(100-counts[0]/fold_size*100)}')
                else:
                    print(f"\nFold {split}:")
                    print(f'Class balance: {round(counts[1]/fold_size*100)}-{round(100-counts[1]/fold_size*100)}')
            
            # Fit classifier
            clf.fit(X_train, y_train)
            probas_ = clf.predict_proba(X_test)
    
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            # Append the tpr vs fpr values interpolated over the mean_fpr linspace
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3, 
                    label='ROC fold %d (AUC = %0.2f)' % (split+1, roc_auc))
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        tpr_per_classifier.append(mean_tpr.T)
    
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=name)
        ax.legend(loc="lower right", fontsize = 6) # Leave it at  6 until we agree on how to move forward
        ax.grid(True)
        # Plot chance curve
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        
        values_test, counts_test = np.unique(y[test_index], return_counts=True)
        values_train, counts_train = np.unique(y[test_index], return_counts=True)
        if ii ==0:
            print(f'\nINFO: Class balance in test set (C-P): {round(counts_test[0]/(y[test_index].size)*100)}-{round(counts_test[1]/(y[test_index].size)*100)}')
            print(f'\nINFO: Class balance in training set (C-P): {round(counts_train[0]/(y[test_index].size)*100)}-{round(counts_train[1]/(y[test_index].size)*100)}')
        ii =+1
        print(f'\nClassifier = {clf}')
        print('AUC = %0.2f \u00B1 %0.2f' % (mean_auc, std_auc))
    
    axs[0,0].set(ylabel = 'False Positive Rate')
    axs[1,0].set(ylabel = 'False Positive Rate')
    axs[1,0].set(xlabel = 'True Positive Rate') 
    axs[1,1].set(xlabel = 'True Positive Rate') 
    
    # Add figure title and save it to metadata
    if metadata["scaling"]:
        figure_title = f'Task: {metadata["task"]}, Band type: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]}, Using one-segment: {metadata["one_segment_per_task"]}, Scaling: {metadata["scaling"]}, RobustScaler'
    else:
        figure_title = f'Task: {metadata["task"]}, Band type: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]}, Using one-segment: {metadata["one_segment_per_task"]}, Scaling: {metadata["scaling"]}'
    fig.suptitle(figure_title)
    metadata["Title"] = figure_title
    
    return metadata

def save_figure(metadata):
    """
    Transforms  metadata to string dict
    
    Saves figure to file with metadata (WIP)
    
    Inputs:
        - save_figure : bool
                Defines whether the figure should be saved to disk or not
    Outputs:
    
    """
    # Deefine filename
    if metadata["normalization"] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_normalized_not-scaled.png'
    elif not metadata["normalization"] and metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_scaled.png'
    elif not metadata["normalization "] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_not-scaled.png'
        
    #figures_dir = "." # Delete after testing
    # Save the figure
    metadata_str = {key: str(value) for key, value in metadata.items()}
    plt.savefig(os.path.join(figures_dir, figure_filename), metadata = metadata_str)
    print(f'\nINFO: Success! Figure "{figure_filename}" has been saved to folder {figures_dir}')

def output_results(metadata):
    """
   Creates a report? Or outputs data
    """

if __name__ == "__main__":
    
    #%% Initialize variables and define arguments
    verbosity = False
    
    dataframe, metadata = load_data()
    metadata = initialize_variables(metadata)
    
    # Define classifiers
    classifiers = [
            ('Support Vector Machine', SVC(kernel = 'rbf', probability=True, random_state=metadata["seed"])),
            ('Logistic Regression', LogisticRegression(penalty='l1', solver='liblinear', random_state=metadata["seed"])),
            ('Random Forest', RandomForestClassifier(random_state=metadata["seed"])),
            ('Linear Discriminant Analysis', LinearDiscriminantAnalysis(solver='svd'))
    ]
    # Define input data, initialize CV and get data split
    X, y, groups, data_split = initialize_cv(dataframe, metadata)
    
    # Fit classifiers and plot
    metadata = fit_and_plot(X, y, groups, classifiers, data_split, metadata)
    
    # Save the figure to file
    save_figure(metadata)
    
    # Output results
#%% 




# from PIL import Image, PngImagePlugin

# # Load the image without metadata
# pil_image = Image.open(os.path.join(figures_dir, figure_filename))

# png_info = PngImagePlugin.PngInfo()
# for key, value in metadata_str.items():
#     png_info.add_text(key, value)

# # Save the image with the metadata
# output_filename = "output_papa.png"
# pil_image.save(output_filename, "PNG", pnginfo=png_info)

# #%% Export data from run
# def output_data():
#     # Create / print out the CSV file with the data
#     pass
# def test_image_with_metadata():

#     # Load the PNG image with metadata    
#     figure_test = plt.imread("output_papa.png")
#     pil_image_with_metadata = Image.open("output_papa.png")
#     # Retrieve the metadata from the image
#     metadata_test = PngImagePlugin.PngInfo(pil_image_with_metadata.info).items()    
#     # Print the metadata
#     print(metadata_test)
    
#     # Display the image
#     plt.imshow(figure_test)
#     plt.show()
    
# Info to be added to the metadata
#metadata = {
#        "Creation Time": datetime.now(),
#        "Author": "WIP",
#        "Workstation": "WIP",
#        "License": "project_license", 
#        "Dataset": "k22",
#        "Population of subjects": "WIP",
#        "Class balance": "WIP",
#        "Number of observations used in classification": len(X),
#        "Number of features per observation": X.shape[1],
#        # Per clf:
#        "Sensitivity": "WIP",
#        "Specifictiy":"WIP",
#        }
#
#
#with open("output_data.txt","w") as file:
#    file.write(f'Date and time of running: {datetime.now()}\n')
#    file.write(f'User: WIP\n')
#    file.write(f'Workstation: WIP \n')
#    file.write(f'Data from dataset: k22 WIP')
#    file.write(f'\nTask: {task} \nBandwidth: {bands} \nNumber of folds in CV: {folds}\n')
#    #file.write(f'Classifier model used: {classifier}\n')
#    file.write(f'Other parameters: WIP\n')    
##    file.write(f'**Results:** \nMean accuracy: {accuracy_average} ± {accuracy_std}\nAUC = {AUC_mean} ± {AUC_std}\n')
#    file.write(f'Number of observations used in the classification: {len(X)}\n')
#    file.write(f'Number of features per observation: {X.shape[1]}\n')
#    # Sensitivity
#    # Specificity
#    # Number of controls and patients
#    
    
 
#%%
    
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

# By default, roc_curve it uses as many thresholds as there are unique values in the y_score input array. Here is the relevant excerpt from the scikit-learn documentation:

# ## (Optional)
    # Define what strategy to use based on the subjects balance

# Note: There is a 27% chance that there's a fold with only one class. This can impact the classifier (especially LDA)
# >>> After 999 iterations, we found 267 folds with 1 class
     