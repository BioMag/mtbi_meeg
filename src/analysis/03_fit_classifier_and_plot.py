#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#################################
# 03_fit_classifier_and_plot.py #
#################################

@authors: Verna Heikkinen, Aino Kuusi, Estanislao Porta

Takes the processed data, fits four different ML classifiers,
performs cross validation and evaluates the performance of the classification
using mean ROC curves.

Data is split it in folds according to 10-fold StratifiedGroupKFold
If only one segment of a task is to be used, CV is done using StratifiedKFold CV.
Arguments used to run the script are added to pickle object.

Arguments
---------
    - eeg_tmp_data.pickle : pickle object
        Object of pickle format containing the dataframe with the data
        and the metadata with the information about the arguments
        used to run the 01_read_processed_data script.
    - seed : int
        Value for initialization of the classifiers and the CV.
    - scaling : bool
        Define whether to perform scaling over data or not.
    - scaling_method : str
        Define what is the preferred scaling method.
    - one_segment_per_task : bool
        Define whether one or all segments of the task will be used for the classification.
    - which_segment : int
        Defines which of the segments will be used.
    - dont_save_fig: bool
        Define whether to refrain from saving the figure to disk.
    - display_figure: bool
        Define whether to display the figure in graphical interface
        (e.g., when running script in HPC).

Returns
-------
    - Prints out figure
    - figure : pickle object
        Object of pickle format containing the dataframe with the data
        and the metadata with the information about the arguments
        used to run this script.
    - metadata?
    - report?

# TODO: Define metric and export them to CSV file
"""
import sys
import os
import argparse
import time
from math import sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statistics import mean, stdev

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)
from config_common import figures_dir, reports_dir
from config_eeg import seed, folds
from pickle_data_handler import PickleDataHandler
# Create directory if it doesn't exist
if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir)

def initialize_argparser(metadata):
    """ Initialize argparser and add args to metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', action='store_true', help='Define the verbosity of the output. Default: False', default=False)
    parser.add_argument('-s', '--seed', type=int, help=f'Seed value used for CV splits, and for classifiers and for CV splits. Default: {seed}', metavar='int', default=seed) # Note: different sklearn versions could yield different results 
    parser.add_argument('--scaling', action='store_true', help='Scaling of data before fitting. Can only be used if data is not normalized. Default: False', default=False)
    parser.add_argument('--scaling_method', choices=scaling_methods, help='Method for scaling data, choose from the options. Default: RobustScaler', default=scaling_methods[2]) 
    parser.add_argument('--one_segment_per_task', action='store_true',  help='Utilizes only one of the segments from the tasks. Default: False', default=False)
    parser.add_argument('--which_segment', type=int, help='Define which number of segment to use: 1, 2, etc. Default is 1', metavar='', default=1)
    parser.add_argument('--display_fig', action='store_true', help='Displays the figure. Default: False', default=False)
    parser.add_argument('--dont_save_fig', action='store_true', help='Saves figure to disk. Default: True', default=False)
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()
    
    # Add the input arguments to the metadata dictionary
    metadata["folds"] = folds
    metadata["seed"] = seed
    metadata["verbosity"] = args.verbosity
    if args.scaling and metadata["normalization"]:
        raise TypeError("You are trying to scale data that has been already normalized.")
    metadata["scaling"] = args.scaling
    metadata["scaling_method"] = args.scaling_method
    metadata["one_segment_per_task"] = args.one_segment_per_task
    metadata["which_segment"] = args.which_segment
    if  args.one_segment_per_task and (args.which_segment > metadata["segments"]):
        raise TypeError(f'The segment you chose is larger than the number of available segments for task {metadata["task"]}. Please choose a value between 1 and {metadata["segments"]}.')
    metadata["display_fig"] = args.display_fig

    return metadata, args

def initialize_cv(dataframe, metadata):
    """Initialize Cross Validation and gets data splits as a list """
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

def initialize_subplots(metadata):
    """Creates figure with 2x2 subplots, sets axes and fig title"""
    # Disable interactive mode in case plotting is not needed
    plt.ioff()
    fig_roc, axs = plt.subplots(nrows=2, ncols=2,
                            sharex=True, sharey=True,
                            figsize=(10, 10))

    # Add figure title and save it to metadata
    if metadata["scaling"]:
        figure_title = (
            f'Task: {metadata["task"]}, Freq band: {metadata["freq_band_type"]}, '
            f'Channel data normalization: {metadata["normalization"]}, \n'
            f'Using one-segment: {metadata["one_segment_per_task"]}, Scaling: '
            f'{metadata["scaling"]}, metadata["scaling_method"]'
        )
    else:
        figure_title = (
            f'Task: {metadata["task"]}, Band type: {metadata["freq_band_type"]}, '
            f'Channel data normalization: {metadata["normalization"]}, \n'
            f'Using one-segment: {metadata["one_segment_per_task"]}, Scaling: '
            f'{metadata["scaling"]}'
        )
    fig_roc.suptitle(figure_title)
    # Add x and y labels
    axs[0, 0].set(ylabel='True Positive Rate')
    axs[1, 0].set(ylabel='True Positive Rate')
    axs[1, 0].set(xlabel='False Positive Rate')
    axs[1, 1].set(xlabel='False Positive Rate')

    # Display figure if needed
    if metadata["display_fig"]:
        plt.show(block=False)
    else:
        print('INFO: Figure will not be displayed.')
    return fig_roc, axs, metadata

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

    # Control if there's only one class in a fold
    if np.unique(y[test_index]).size == 1:
        print(f"WARN: Split {split+1} has only 1 class in the test set, skipping it. ####")
        skip_split = True
    # Print out class balance if needed
    if metadata["verbosity"]:
        print(f"\nSplit {split+1}:")
        _, counts_test = np.unique(y[test_index], return_counts=True)
        _, counts_train = np.unique(y[train_index], return_counts=True)

        print(f'INFO: Class balance in test set (C-P): '
              f'{round(counts_test[0]/(y[test_index].size)*100)}-'
              f'{round(counts_test[1]/(y[test_index].size)*100)}')
        print(f'INFO: Class balance in training set (C-P): '
              f'{round(counts_train[0]/(y[train_index].size)*100)}-'
              f'{round(counts_train[1]/(y[train_index].size)*100)}')

    return X_train, X_test, y_train, y_test, skip_split

def roc_per_clf(tprs, aucs, ax, name, clf):
    """ Calculates the mean TruePositiveRate and AUC for classifier 'clf'.
    Adds confidence interval of the AUC to the figure
    Adds the chance plot to the figure
    """
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # Calculate AUC's mean and confidence interval based on fpr and tpr and add to plot
    mean_auc = round(auc(mean_fpr, mean_tpr), 3)
    std_auc = round(np.std(aucs), 3)
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=r'AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    # Calculate upper and lower std_dev band around mean and add to plot
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='tab:blue', alpha=.2)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='tab:grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[0, 1], ylim=[0, 1], title=name)
    ax.legend(loc="lower right", fontsize=12) 
    #ax.grid(True)
    # Plot chance curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='tab:red',
            label='Chance', alpha=.3)
    # Estimate confidence interval
    ci_auc = round(std_auc*1.96/sqrt(folds), 3)
    print(f'\nINFO: Classifier = {clf}')
    print('\tAUC = %0.2f \u00B1 %0.2f' % (mean_auc, ci_auc))

    return mean_tpr

def metrics_per_clf(sensitivity, specificity, accuracy):
    """Calculates metrics and confidence interval for each classifier"""
    mean_sens = round(mean(sensitivity), 2)
    ci_sens = round(stdev(sensitivity)*1.96/sqrt(folds), 2)
    mean_spec = round(mean(specificity), 2)
    ci_spec = round(stdev(specificity)*1.96/sqrt(folds), 2)
    mean_acc = round(mean(accuracy), 2)
    ci_acc = round(stdev(accuracy)*1.96/sqrt(folds), 2)

    print('\tSensitivity = %0.2f \u00B1 %0.2f' % (mean_sens, ci_sens))
    print('\tSpecificity = %0.2f \u00B1 %0.2f' % (mean_spec, ci_spec))
    print('\tAccuracy = %0.2f \u00B1 %0.2f' % (mean_acc, ci_acc))
    return mean_sens, ci_sens, mean_spec, ci_spec, mean_acc, ci_acc

def fit_and_plot(X, y, classifiers, data_split, metadata):
    """
    Loops over all classifiers according to the data split of the CV.
    Plots the split ROCs in subplots
    Arguments
    ---------
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

    Returns
    -------
         - Figure with 2x2 subplots: matplotlib plot
         - metadata : dict containing df 'metrics', which includes:
                - tpr_per_classifier : list
                - sensitivity_per_classifier : list
                - specificity_per_classifier : list
                - f1_per_classifier : list
    """
    # Initialize dataframe where the metrics will be stored
    tpr_per_classifier = []
    accuracy_per_classifier = []
    ci_acc_clf = []
    sensitivity_per_classifier = []
    ci_sens_clf = []
    specificity_per_classifier = []
    ci_spec_clf = []
    # Submethod 4.1 - Initialize the subplots
    fig_roc, axs, metadata = initialize_subplots(metadata)
    # Iterate over the classifiers to populate each subplot
    for ax, (name, clf) in zip(axs.flat, classifiers):
        tprs = []
        aucs = []
        accuracy = []
        sensitivity = []
        specificity = []
        mean_fpr = np.linspace(0, 1, 100)
        # Fit the classifiers to the split
        for split, (train_index, test_index) in enumerate(data_split):
            # Submethod 4.2 - Slice the X and y data according to CV's data_split
            X_train, X_test, y_train, y_test, skip_split = \
                perform_data_split(X, y, split, train_index, test_index)
            # Skip this split if class balance is bad
            if skip_split:
                continue

            # Fit classifier and predict outcomes
            clf.fit(X_train, y_train)
            probas_ = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
            # Append the (tpr vs fpr) values interpolated over mean_fpr
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Calculate the AUC for this ROC
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # Plot the ROC for this split
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label=f'Split {split+1} (AUC = {roc_auc:.2f})')
            # To not add the split AUC to legend, uncomment this: 
            #ax.plot(fpr, tpr, lw=1, alpha=0.3)
            
            # Get the sensitivity, specificity and accuracy values
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() 
            sensitivity.append(tp / (tp + fn))
            specificity.append(tn / (tn + fp))
            accuracy_test = accuracy_score(y_test, y_pred)
            accuracy.append(accuracy_test)
            
            # Print out set's accuracy score to evaluate overfitting:
            if metadata["verbosity"]:
                print(f"INFO: Accuracy score in test set: {accuracy:.2f}")
                y_train_pred = clf.predict(X_train)
                accuracy_train = accuracy_score(y_train, y_train_pred)
                print(f"INFO: Accuracy score in training set: {accuracy_train:.2f}")
            
        # Submethods 4.3 & 4.4 - Calculate means & metrics per classifier
        mean_tpr = roc_per_clf(tprs, aucs, ax, name, clf)
        mean_sensitivity, ci_sens, mean_specificity, ci_spec, mean_accuracy, ci_acc = metrics_per_clf(sensitivity, specificity, accuracy)
        
        tpr_per_classifier.append(mean_tpr.T)
        accuracy_per_classifier.append(mean_accuracy)
        ci_acc_clf.append(ci_acc)
        sensitivity_per_classifier.append(mean_sensitivity)
        ci_sens_clf.append(ci_sens)
        specificity_per_classifier.append(mean_specificity)
        ci_spec_clf.append(ci_spec)

    metrics = pd.DataFrame({
                    'Classifiers': [pair[0] for pair in classifiers],
                    'Accuracy': accuracy_per_classifier,
                    'Accuracy_CI': ci_acc_clf,
                    'Sensitivity': sensitivity_per_classifier,
                    'Sensitivity_CI': ci_sens_clf,
                    'Specificity': specificity_per_classifier,
                    'Specificity_CI': ci_spec_clf,
                    'TPR': tpr_per_classifier
                    })
    metadata["metrics"] = metrics
    return fig_roc, metadata

def plot_boxplot(metadata):
    """Plot boxplot of mean AUC, Sensitivity and Specificity and their 95% confidence intervals"""
    df = metadata["metrics"]
   # Set up marker shapes and colors for each classifier
    markers = ["o", "^", "s", "d"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # Create subplots for each metric
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    # Iterate over each metric
    for i, metric in enumerate(['Accuracy', 'Sensitivity', 'Specificity']):
        ax = axs[i]
            
        # Iterate over each classifier
        for j, clf in enumerate(df.index):
            # Get the mean value and CI of the current metric for the current classifier
            mean_val = df.loc[clf, metric]
            ci = df.loc[clf, metric+'_CI']           
            # Plot the mean value as a marker with the corresponding shape and color
            ax.plot(j, mean_val, marker=markers[j], markersize=10, color=colors[j])
            # Plot the CI as a vertical error bar
            ax.vlines(j, mean_val - ci, mean_val + ci, color=colors[j], linewidth=2)
            
        # Set the x-axis tick labels to be the classifier names
        ax.set_xticks(np.arange(len(df.index)))
        ax.set_xticklabels([])
        ax.set_ylim(0,1)
        # Set the y-axis label to be the current metric name
        ax.set_ylabel(metric)
        ax.axhline(y=0.5, color='grey', linestyle='--')
    # Add a legend for the marker shapes and the corresponding classifiers
    handles = []
    for j, clf in enumerate(df["Classifiers"]):
        handle = plt.Line2D([0], [0], marker=markers[j], color='w', label=clf, markerfacecolor=colors[j], markersize=10)
        handles.append(handle)
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(classifiers))
    
    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle("Classification metrics")
    if metadata["display_fig"]:
        plt.show(block=False)
    else:
        print('INFO: Figure will not be displayed.')
    return fig


def save_figures(metadata):
    """Saves active  figure to disk"""
    # Define filename
    if metadata["normalization"] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_normalized_not-scaled.png'
    elif not metadata["normalization"] and metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_scaled.png'
    elif not metadata["normalization"] and not metadata["scaling"]:
        figure_filename = f'{metadata["task"]}_{metadata["freq_band_type"]}_not-normalized_not-scaled.png'

    # Save the figure
    metadata["roc-plots-filename"] = figure_filename
    fig_roc.savefig(os.path.join(figures_dir, figure_filename))
    boxplot_filename = f'{metadata["roc-plots-filename"][:-4]}_boxplot.png'
    fig_boxplot.savefig(os.path.join(figures_dir, boxplot_filename))
    
    print(f'INFO: Figures "{figure_filename}" and "{boxplot_filename}" have been saved to folder {figures_dir}')
    
def save_csv(metadata):
    """Saves the classification metrics as a csv"""
    csv_filename = f'{metadata["roc-plots-filename"][:-4]}.csv'
    csv_path = os.path.join(reports_dir, csv_filename)
    df = metadata["metrics"]
    with open(csv_path, 'w') as file:
        file.write(f'#{metadata["timestamp"]}\n')
        df.to_csv(file, index=False)
    print(f'INFO: CSV data with metrics "{csv_filename}" has been saved to folder {figures_dir}')

if __name__ == "__main__":

    # Save time of beginning of the execution to measure running time
    start_time = time.time()

    # 1 - Read data
    handler = PickleDataHandler()
    dataframe, metadata = handler.load_data()

    # Define scaling methods and classifiers
    scaling_methods = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    classifiers = [
        ('Support Vector Machine', SVC(kernel='rbf', probability=True, random_state=seed)),
        ('Logistic Regression', LogisticRegression(penalty='l1', solver='liblinear', random_state=seed)),
        ('Random Forest', RandomForestClassifier(random_state=seed)),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis(solver='svd'))
    ]
    metadata["Classifiers"] = classifiers

    # 2 - Initialize command line arguments and save arguments to metadata
    metadata, args = initialize_argparser(metadata)

    # 3 - Define input data, initialize CV and get data split
    X, y, data_split = initialize_cv(dataframe, metadata)

    # 4 - Fit classifiers and plot
    fig_roc, metadata = fit_and_plot(X, y, classifiers, data_split, metadata)
    # 4.5 - Plot  boxplot
    fig_boxplot = plot_boxplot(metadata)
    
    # 5 -  Add timestamp
    metadata["timestamp"] = datetime.now()

    # 6 - Save the figure to disk
    if args.dont_save_fig:
        print('INFO: Figures will not be saved to disk.')
    else:
        save_figures(metadata)
    
    # 7 - Save CSV data to reports dir
    save_csv(metadata)
    
    # 8 - Export metadata
    handler.export_data(dataframe, metadata)

    # Calculate time that the script takes to run
    execution_time = (time.time() - start_time)
    print('\n###################################################\n')
    print(f'Execution time of 03_fit_classifier_and_plot.py: {round(execution_time, 2)} seconds\n')
    print('###################################################\n')
