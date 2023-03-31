#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#################################
# 03_fit_classifier_and_plot.py #
#################################

@authors: Verna Heikkinen, Aino Kuusi, Estanislao Porta

Takes the processed data, fits four different ML classifiers, performs cross validation and evaluates the performance of the classification using mean ROC curves.

Data is split it in folds according to 10-fold StratifiedGroupKFold (when using all segments of a task). If only one segment of a task is to be used, CV is done using StratifiedKFold CV. Arguments used to run the script are added to pickle object.


Arguments
---------
    - eeg_tmp_data.pickle : pickle object
        Object of pickle format containing the dataframe with the data and the metadata with the information about the arguments used to run the 01_read_processed_data script.           
    - seed : int      
    - scaling : bool
    - scaling_method : str
    - one_segment_per_task : bool
        whether one or all segments of the task will be used
    - which_segment : int
        If one_segment_per_task is True, this defines which of the segments will be used
        
Returns
-------
    - Prints out figure
    - figure : pickle object 
        Object of pickle format containing the dataframe with the data and the metadata with the information about the arguments used to run this script.
    - metadata?
    - report?

# TODO: Define what to do with metrics: save in metadata / export a CSV file
# TODO: Add logging?
# TODO: Add stuff to metadata as per the comments below: possibly not in this script?
"""
import sys
import os
import argparse
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statistics import mean, stdev

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)
from config_common import figures_dir
from config_eeg import seed, folds
from pickle_data_handler import PickleDataHandler
# Create directory if it doesn't exist
if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir)


def initialize_subplots(metadata):
    """Creates figure with 2x2 subplots, sets axes and fig title"""
    fig, axs = plt.subplots(nrows=2, ncols=2, 
                            sharex=True, sharey=True, 
                            figsize=(10, 10))
    # Add figure title and save it to metadata
    if metadata["scaling"]:
        figure_title = f'Task: {metadata["task"]}, Freq band: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]}, \nUsing one-segment: {metadata["one_segment_per_task"]}, Scaling: {metadata["scaling"]}, RobustScaler'
    else:
        figure_title = f'Task: {metadata["task"]}, Band type: {metadata["freq_band_type"]}, Channel data normalization: {metadata["normalization"]}, \nUsing one-segment: {metadata["one_segment_per_task"]}, Scaling: {metadata["scaling"]}'
    fig.suptitle(figure_title)
    # Add x and y labels
    axs[0, 0].set(ylabel='False Positive Rate')
    axs[1, 0].set(ylabel='False Positive Rate')
    axs[1, 0].set(xlabel='True Positive Rate') 
    axs[1, 1].set(xlabel='True Positive Rate')     
    # Disable interactive mode in case plotting is not needed
    plt.ioff()    
    # Display figure if needed    
    if metadata["display_fig"]:
        plt.show()
    
    return axs, metadata

def perform_data_split(X, y, split, train_index, test_index):
    """Splits X and y data into training and testing according to the data split indexes"""
    skip_split = False
    # Generate train and test sets for this split
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Scale if needed:
    if metadata["scaling"] and not metadata["normalization"]:
        scaler = metadata["scaling_method"]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif metadata["scaling"] and metadata["normalization"]:
        raise TypeError("You are trying to scale data that has been already normalized.")
    # Control if there's only one class in a fold
    if np.unique(y[test_index]).size == 1:
        print(f"WARN: Split {split+1} has only 1 class in the test set, skipping it. ####")
        skip_split = True
    # Print out class balance if needed 
    if metadata["verbosity"]: 
        print(f"\nSplit {split+1}:")            
        _, counts_test = np.unique(y[test_index], return_counts=True)
        _, counts_train = np.unique(y[train_index], return_counts=True)
    
        print(f'INFO: Class balance in test set (C-P): {round(counts_test[0]/(y[test_index].size)*100)}-{round(counts_test[1]/(y[test_index].size)*100)}')
        print(f'INFO: Class balance in training set (C-P): {round(counts_train[0]/(y[train_index].size)*100)}-{round(counts_train[1]/(y[train_index].size)*100)}')        
    
    return X_train, X_test, y_train, y_test, skip_split

def roc_per_clf(tprs, aucs, ax, name, clf):
    """ Calculates the mean TruePositiveRate and AUC for classifier 'clf'"""
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # Calculate AUC's mean and std_dev based on fpr and tpr and add to plot
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    # Calculate upper and lower std_dev band around mean and add to plot
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=name)
    ax.legend(loc="lower right", fontsize=6) # Leave it at  6 until we agree on how to move forward
    ax.grid(True)
    # Plot chance curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    print(f'\nINFO: Classifier = {clf}')
    print('\tAUC = %0.2f \u00B1 %0.2f' % (mean_auc, std_auc))
    
    return mean_tpr, mean_auc, std_auc

def metrics_per_clf(precision, recall, f1):
    mean_precision = mean(precision)
    std_precision = stdev(precision)
    mean_recall = mean(recall)
    std_recall = stdev(recall)
    mean_f1 = mean(f1)
    std_f1 = stdev(f1)
    
    print('\tPrecision = %0.2f \u00B1 %0.2f' % (mean_precision, std_precision))
    print('\tRecall = %0.2f \u00B1 %0.2f' % (mean_recall, std_recall))
    print('\tF1 = %0.2f \u00B1 %0.2f' % (mean_f1, std_f1))
    
    return mean_precision, mean_recall, mean_f1

def initialize_cv(dataframe, metadata):
    """
    Initialize Cross Validation and get data splits as a list
    return X, y, groups, metadata, data_split
    """
    # Define features, classes and groups
    X = dataframe.iloc[:, 2:]
    y = dataframe.loc[:, 'Group']
    groups = dataframe.loc[:, 'Subject']
    
    # Slice data
    if metadata["one_segment_per_task"]:
        # Removes (segments-1) rows out of the dataframe 
        X = X[metadata["which_segment"]:len(X):metadata["segments"]]
        y = y[metadata["which_segment"]:len(y):metadata["segments"]]
        groups = groups[metadata["which_segment"]:len(groups):metadata["segments"]]
        
        # Initialize Stratified K Fold
        skf = StratifiedKFold(n_splits=metadata["folds"], shuffle=True, random_state=seed)
        data_split = list(skf.split(X, y, groups))
    else:
        # Initialize Stratified Group K Fold
        sgkf = StratifiedGroupKFold(n_splits=metadata["folds"], shuffle=True, random_state=seed)
        data_split = list(sgkf.split(X, y, groups))

    return X, y, data_split

def fit_and_plot(X, y, classifiers, data_split, metadata):
    """
    Loops over all classifiers according to the data split of the CV.
    Plots the results in subplots
    Parameters
    ----------
        - X : list
            Sample subjects
        - y : list
            Features of the samples
        - classifiers :  list
            List with the functions used as ML classifiers
        - data_split : list
            Indexes  of the Training and Testing sets for the CV splits
        - metadata : dict
            Object containing the parameters used in the analysis
            
    returns
    -------
         - Figure with 2x2 subplots: matplotlib plot
         - metadata : dict
         - tpr_per_classifier : list
         - precision_per_classifier : list
         - recall_per_classifier : list
         - F1_per_classifier : list
    """
    tpr_per_classifier = []
    auc_per_classifier = []
    std_auc_per_classifier  = []
    
    # Initialize the subplots
    axs, metadata = initialize_subplots(metadata)       
    # Iterate over the classifiers to populate each subplot
    for ax, (name, clf) in zip(axs.flat, classifiers):
        tprs = []
        aucs = []
        f1 = []
        precision = []
        recall = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Fit the classifiers to get the data that will be used in subplots
        for split, (train_index, test_index) in enumerate(data_split):
            # Slice the X and y data according to the data_split
            X_train, X_test, y_train, y_test, skip_split = \
                perform_data_split(X, y, split, train_index, test_index)
            
            if skip_split:
                continue
            
            # Fit classifier
            clf.fit(X_train, y_train)
            # Predict outcomes
            probas_ = clf.predict_proba(X_test) 
            y_pred = clf.predict(X_test)
            # Compare predicted and compute ROC curve 
            fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
            # Append the (tpr vs fpr) values interpolated over mean_fpr
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # Plot the ROC for this split
            ax.plot(fpr, tpr, lw=1, alpha=0.3, 
                    label='ROC split %d (AUC = %0.2f)' % (split+1, roc_auc))
            # Append the precision, recall and F1 values
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
          
        # Execute the functions that calculate means & metrics per classifier    
        mean_tpr, mean_auc, std_auc = roc_per_clf(tprs, aucs, ax, name, clf)
        mean_precision, mean_recall, mean_f1  = metrics_per_clf(precision, recall, f1)
    
        tpr_per_classifier.append(mean_tpr.T)
        auc_per_classifier.append(mean_auc.T)
        std_auc_per_classifier.append(std_auc.T)
        # Add precision, recall, F1 per classifier?
    return metadata # TODO: return something else here?

def add_timestamp(metadata):
    """Adds timestamp to metadata"""
    metadata["timestamp"] = datetime.now()
    
    return metadata

def save_figure(metadata):
    """    
    Saves ACTIVE figure to disk
    Input parameters:
    -----------------
        - save_figure : bool WIP
                Defines whether the figure should be saved to disk or not    
    """
    # Define filename
    if metadata["normalization"] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_normalized_not-scaled.png'
    elif not metadata["normalization"] and metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_scaled.png'
    elif not metadata["normalization"] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_not-scaled.png'
 
    # Save the figure
    metadata["roc-plots-filename"] = figure_filename
    plt.savefig(os.path.join(figures_dir, figure_filename))
    print(f'\nINFO: Success! Figure "{figure_filename}" has been saved to folder {figures_dir}')


if __name__ == "__main__":
    
    # Save time of beginning of the execution to measure running time
    start_time = time.time()
    
    # Scaling methods
    scaling_methods = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    # Define classifiers
    classifiers = [
        ('Support Vector Machine', SVC(kernel='rbf', probability=True, random_state=seed)),
        ('Logistic Regression', LogisticRegression(penalty='l1', solver='liblinear', random_state=seed)),
        ('Random Forest', RandomForestClassifier(random_state=seed)),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis(solver='svd'))
    ]
       
    # Add arguments to be parsed from command line    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', type=bool, help="Define the verbosity of the output. Default is False", metavar='', default=True)
    parser.add_argument('-s', '--seed', type=int, help=f"Seed value used for CV splits, and for classifiers and for CV splits. Default value is {seed}, and gives 50/50 class balance in Training and Test sets.", metavar='int', default=seed) # Note: different sklearn versions could yield different results
    parser.add_argument('--scaling', type=bool, help='Scaling of data before fitting. Can only be used if data is not normalized. Default is True', metavar='', default=False)
    parser.add_argument('--scaling_method', choices=scaling_methods, help='Method for scaling data, choose from the options. Default is RobustScaler.', default=scaling_methods[2])
    parser.add_argument('--one_segment_per_task', type=bool, help='Utilize only one of the segments from the tasks. Default is False', metavar='', default=False)
    parser.add_argument('--which_segment', type=int, help='Define which number of segment to use: 1, 2, etc. Default is 1', metavar='', default=1)
    parser.add_argument('--display_fig', type=int, help='Define whether figure will be shown. Default is true', metavar='', default=True)
    parser.add_argument('--save_fig', type=int, help='Define whether figure will be saved. Default is true', metavar='', default=False)    
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()
    
    # Execute the submethods:
    # 1 - Read data
    handler = PickleDataHandler()
    dataframe, metadata = handler.load_data()
    
    # 2 - Add the input arguments to the metadata dictionary
    metadata["folds"] = folds
    metadata["seed"] = seed
    metadata["verbosity"] = args.verbosity
    metadata["scaling"] = args.scaling
    metadata["scaling_method"] = args.scaling_method
    metadata["one_segment_per_task"] = args.one_segment_per_task
    metadata["display_fig"] = args.display_fig
    # Which segment to be used when using only one segment for fitting 
    if  args.one_segment_per_task:
        metadata["which_segment"] = (args.which_segment)
    elif args.which_segment > metadata["segments"]:
        raise TypeError(f'The segment you chose is larger than the number of available segments for task {metadata["task"]}. Please choose a value between 1 and {metadata["segments"]}.')
             
    print(f'INFO: Input parameters: \n\tTask: {metadata["task"]}, \n\tBand type: {metadata["freq_band_type"]}, \n\tChannel data normalization: {metadata["normalization"]}, \n\tUsing one-segment: {metadata["one_segment_per_task"]}, \n\tScaling: {metadata["scaling"]}, \n\tScaler method: {metadata["scaling_method"]} \n\nINFO: Data is being split and fitted, please wait a moment... \n')
    
    # 3 - Define input data, initialize CV and get data split
    X, y, data_split = initialize_cv(dataframe, metadata)
    
    # 4 - Fit classifiers and plot
    metadata = fit_and_plot(X, y, classifiers, data_split, metadata)
    
    # 5 -  Add timestamp
    metadata = add_timestamp(metadata)
    
    # 6 - Save the figure to disk
    if args.save_fig:
        save_figure(metadata)
    else:
        print('INFO: Figure will not be saved to disk')
    
    # 7 - Export metadata
    handler.export_data(dataframe, metadata)
    
    # Calculate time that the script takes to run
    execution_time = (time.time() - start_time)
    print('\n###################################################\n')
    print(f'Execution time of 03_fit_classifier_and_plot.py: {round(execution_time, 2)} seconds\n')
    print('###################################################\n')


# Info to be added to the metadata
#metadata = {
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
