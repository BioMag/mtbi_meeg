#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:01:16 2022

@author: aino

Tests the performance of different models (logistic regression (lr), linear discriminant analysis (lda),
support vector machine (svm), neighborhood component analysis (nca), quadratic discriminant analysis (qda)
and random forests (rf)) using stratified k fold cross validation. Plots confusion matrices for these models.
Plots ROC curve for chosen model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
import numpy as np
#from readdata import dataframe


import pandas as pd
import matplotlib.pyplot as plt
import math



def one_split(clfs, X, y, confusion_m, pca, feature_importance):
    """
    Tests the performance of chosen models. 

    Parameters
    ----------
    clfs : list
        List of classifiers to train 
    X : dataframe
        Vectoriced PSDs (features) of each subject (observation)
    y : series
        Classification labels (binary)
    confusion_m : bool
        If True, will plot confusion matrices
    pca : bool
        If True, performs PCA before proceeding with the analysis
    feature_importance : bool
        If True, will plot feature importances for RandomForest classifier.
        Ignored for other classifiers.

    Returns
    -------
    None.

    """
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=20)
    classifiers = []
    # Dimensionality reduction with PCA
    if pca: 
        pca = PCA(n_components=4)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)
       
    # Logistic regression
    if 'lr' in clfs:
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)  
        print(clf.score(X_test, y_test))
        classifiers.append(clf)
    
    # Linear discriminant analysis
    if 'lda' in clfs:
        clf_2 = LinearDiscriminantAnalysis().fit(X_train, y_train)
        print(clf_2.score(X_test, y_test))
        classifiers.append(clf_2)
    
    # Support vector machine
    if 'svm' in clfs:
        clf_3 = svm.SVC()
        clf_3.fit(X_train, y_train)
        print(clf_3.score(X_test, y_test))
        classifiers.append(clf_3)
    
    # Neighborhood component analysis
    if 'nca' in clfs:
        nca = NeighborhoodComponentsAnalysis()
        knn = KNeighborsClassifier()
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, y_train)
        print(nca_pipe.score(X_test, y_test))
        # Random forest
    if 'rf' in clfs:
        clf_rf = RandomForestClassifier().fit(X_train, y_train)
        classifiers.append(clf_rf)
        if feature_importance:
            plot_feature_importance(clf_rf)
            
    # if confusion_m:
    #     plot_confusion_matrices(len(clfs), classifiers, X_test, y_test, clfs)
        
    

def plot_confusion_matrices(dim, clfs, X_test, y_test, titles):
    """
    Plots confusion matrices for chosen classifiers.
    
    Parameters
    ----------
    dim : int
        The number of classifiers i.e. the number of subplots
    clfs : list
        List of classifiers 
    X_test : dataframe
        Subset of the data that is used to test the performance of the classifiers (features: vectorized PSDs, observations: subjects)
    y_test : dataframe
        Classification labels for X_test (binary)
    titles : str
        List of the names of the classifiers for captions

    Returns
    -------
    None.

    """
    
    figure, axis = plt.subplots(1, dim)
    for clf, ax, title in zip(clfs, axis.flatten(), titles):
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
        ax.title.set_text(title)
    plt.tight_layout()
    plt.show()


    

def plot_feature_importance(forest):
    """
    Plots feature importances for n most significant features for RandomForestClassifier

    Parameters
    ----------
    forest : 
        StandardScaler + RandomForestClassifier
    Returns
    -------
    None.

    """
    # Make a list of feature names (this could be moved to readdata)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    channels = [x for x in range(64)]
    bands_and_channels = [(x, y) for x in bands for y in channels]
    feature_names = [f"{x}_{y}" for (x,y) in bands_and_channels]
    
    # Get feature importances
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    sorted_fi = forest_importances.sort_values(ascending=False, axis=0)
    forest_importances_1 = sorted_fi[0:30]
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    std_series = pd.Series(std, index=feature_names)
    indices = forest_importances_1.index
    std_1 = std_series[indices]
    
    # Plot feature importances
    fig, ax = plt.subplots()
    forest_importances_1.plot.bar(yerr=std_1, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    
    
    

    #TODO: fix confusion matrix problem 
def stratified_k_fold_cv(clfs, X, y, folds, confusion_m, pca, feature_importance, standard_scaler):
    """
    Tests the performance of chosen models using stratified k fold cross validation.

    Parameters
    ----------
    clfs : list
        List of classifiers to test
    X : dataframe
        Vectorized PSDs (features) of each subject (observation)
    y : series
        Classification labels (binary)
    folds : int
        The number of folds in the CV
    confusion_m : bool
        If True, plots confusion matrices for the models
    pca : bool
        If True, performs PCA before training the models
    feature_importance : bool
        If True, plots feature importances for RandomForest
        Ignored for other classifiers
    standard_scaler: bool
        If True, uses StandardScaler to normalize the features before training

    Returns
    -------
    scores : list
        List of scores for each classifier and each fold

    """
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    split = skf.split(X, y)
    
    # Lists for scores
    stratified_accuracy_lr =[]
    stratified_accuracy_lda =[]
    stratified_accuracy_svm = []
    stratified_accuracy_nca = []
    stratified_accuracy_qda= []
    stratified_accuracy_rf = []
    
    if standard_scaler:
        scaler = StandardScaler()
    else:
        scaler = None

    
    for train_index, test_index in split:
        # # Get training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        classifiers = []
        # Dimensionality reduction using PCA
        if pca:
            pca = PCA(n_components=15)
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)

        # Logistic regression
        if 'lr' in clfs:
            clf_lr = make_pipeline(scaler, LogisticRegression(penalty='l1',solver='liblinear',random_state=0)).fit(X_train, y_train)
            stratified_accuracy_lr.append(clf_lr.score(X_test, y_test))
            classifiers.append(clf_lr)
            
        # Linear discriminant analysis
        if 'lda' in clfs:
            clf_lda = make_pipeline(scaler, LinearDiscriminantAnalysis(solver='svd')).fit(X_train, y_train)
            stratified_accuracy_lda.append(clf_lda.score(X_test, y_test))
            classifiers.append(clf_lda)
             
        # Support vector machine
        if 'svm' in clfs:
            clf_svm = make_pipeline(scaler, svm.SVC(probability=True))
            clf_svm.fit(X_train, y_train)
            stratified_accuracy_svm.append(clf_svm.score(X_test, y_test))
            classifiers.append(clf_svm)
        
        # Neighborhood component analysis 
        if 'nca' in clfs:
            clf_nca = make_pipeline(scaler, Pipeline([('nca', NeighborhoodComponentsAnalysis()), ('knn', KNeighborsClassifier())]))
            clf_nca.fit(X_train, y_train)
            stratified_accuracy_nca.append(clf_nca.score(X_test, y_test))
            classifiers.append(clf_nca)
            
        # Quadratic discriminant analysis
        if 'qda' in clfs:
            clf_qda = make_pipeline(scaler, QuadraticDiscriminantAnalysis())
            clf_qda.fit(X_train, y_train)
            stratified_accuracy_qda.append(clf_qda.score(X_test, y_test))
            classifiers.append(clf_qda)
            
            ### Warning: variables are collinear (also with PCA)
        # Random forest
        if 'rf' in clfs:
            clf_rf = make_pipeline(scaler, RandomForestClassifier()).fit(X_train, y_train)
            stratified_accuracy_rf.append(clf_rf.score(X_test, y_test))
            classifiers.append(clf_rf)
            # Plot feature importances
            if feature_importance:
                plot_feature_importance(clf_rf)
            
        # Plot confusion matrices
        if confusion_m:
            if len(clfs) >1:
                plot_confusion_matrices(len(clfs), classifiers, X_test, y_test, clfs)
            else:
                plot_confusion_matrices(2, classifiers, X_test, y_test, clfs)
    # Return scores
    scores = []
    for i in [stratified_accuracy_lr, stratified_accuracy_lda, stratified_accuracy_svm, 
              stratified_accuracy_nca, stratified_accuracy_qda, stratified_accuracy_rf]:
        if len(i) != 0:
            scores.append(i)
    return scores


        


if __name__ == "__main__":

    #data = dataframe
    data = pd.read_csv('dataframe.csv', index_col = 'Index')
    X = data.iloc[:, 2:]
    y = data.loc[:, 'Group']
    
    # Choose parameters
    test = 'cv' # How to split the data ('split', 'cv')
    clfs = ['lr','lda', 'svm', 'rf'] # List of models to test ('lr', 'lda', 'svm', 'nca', 'qda', 'rf') !!!IN THIS ORDER!!!
    pca = False
    confusion_m = False
    feature_importance = False
    standard_scaler = True
    folds = 10 
    
    
    # Do we use 'one_split'? I will assume this is not so widely used for now
    if test == 'split':
        one_split(clfs, X, y, confusion_m, pca, feature_importance)
    # Stratified k fold cross validation
    elif test == 'cv':
        scores = stratified_k_fold_cv(clfs, X, y, folds, confusion_m, pca, feature_importance, standard_scaler)
        print(scores)
        for i in scores:
            print(np.mean(i))

# COnfusion matrix prints all black
# Feature importance is not working: "AttributeError: 'Pipeline' object has no attribute 'feature_importances_'"
# What does standard_scaler do? Standard from from sklearn.preprocessing - Standardize features by removing the mean and scaling to unit variance.

