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
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, auc, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
import numpy as np
from readdata import dataframe


import pandas as pd
import matplotlib.pyplot as plt
import math



def one_split(clfs, X, y, confusion_m, pca, feature_importance):
    """

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
        Ignored for other cladsifiers.

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
    figure, axis = plt.subplots(1, dim)
    for clf, ax, title in zip(clfs, axis.flatten(), titles):
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
        ax.title.set_text(title)
    plt.tight_layout()
    plt.show()


    

def plot_feature_importance(forest):
    # Make a list of feature names
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
def stratified_k_fold_cv(clfs, X, y, folds, confusion_m, pca, feature_importance):
    # Stratified K fold cross validation, testing different models
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    split = skf.split(X, y)
    
    # Lists for scores
    stratified_accuracy_lr =[]
    stratified_accuracy_lda =[]
    stratified_accuracy_svm = []
    stratified_accuracy_nca = []
    stratified_accuracy_qda= []
    stratified_accuracy_rf = []
    
    
    
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
            clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1',solver='liblinear',random_state=0)).fit(X_train, y_train)
            stratified_accuracy_lr.append(clf.score(X_test, y_test))
            classifiers.append(clf)
            
        # Linear discriminant analysis
        if 'lda' in clfs:
            clf_2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='svd')).fit(X_train, y_train)
            stratified_accuracy_lda.append(clf_2.score(X_test, y_test))
            classifiers.append(clf_2)
             
        # Support vector machine
        if 'svm' in clfs:
            clf_3 = make_pipeline(StandardScaler(), svm.SVC(probability=True))
            clf_3.fit(X_train, y_train)
            stratified_accuracy_svm.append(clf_3.score(X_test, y_test))
            classifiers.append(clf_3)
        
        # Neighborhood component analysis 
        if 'nca' in clfs:
            nca = NeighborhoodComponentsAnalysis()
            knn = KNeighborsClassifier()
            nca_pipe = make_pipeline(StandardScaler(), Pipeline([('nca', nca), ('knn', knn)]))
            nca_pipe.fit(X_train, y_train)
            stratified_accuracy_nca.append(nca_pipe.score(X_test, y_test))
            classifiers.append(nca_pipe)
            
        # Quadratic discriminant analysis
        if 'qda' in clfs:
            clf_qda = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
            clf_qda.fit(X_train, y_train)
            stratified_accuracy_qda.append(clf_qda.score(X_test, y_test))
            classifiers.append(clf_qda)
            
            ### Warning: variables are collinear (also with PCA)
        # Random forest
        if 'rf' in clfs:
            clf_rf = make_pipeline(StandardScaler(), RandomForestClassifier()).fit(X_train, y_train)
            stratified_accuracy_rf.append(clf_rf.score(X_test, y_test))
            classifiers.append(clf_rf)
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

    data = dataframe
    X, y = data.iloc[:,1:data.shape[1]], data.loc[:, 'Group']
    
    # Choose parameters
    test = 'cv' # How to test the models ('split', 'roc', 'cv')
    clfs = ['lr','lda', 'svm', 'rf'] # List of models to test ('lr', 'lda', 'svm', 'nca', 'qda', 'rf') !!!IN THIS ORDER!!!
    pca = False
    confusion_m = True
    feature_importance = False
    folds = 10
    
    
    
    if test == 'split':
        one_split(clfs, X, y, confusion_m, pca, feature_importance)
    # Stratified k fold cross validation
    elif test == 'cv':
        scores = stratified_k_fold_cv(clfs, X, y, folds, confusion_m, pca, feature_importance)
        print(scores)
        for i in scores:
            print(np.mean(i))
    # Plots a ROC curve and uses stratified k fold cross validation (len(clfs)=1)
    elif test == 'roc':
        print(plot_roc_curve(clfs, X, y, folds))