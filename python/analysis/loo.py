#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:26:43 2022

@author: aino

Tests the performance of three models (logistic regression, linear discriminant 
analysis and support vector machine) using leave one out cross validation.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from readdata import data_frame
from freq_waves import all_dataframe as bands_dataframe

# Get data
data = bands_dataframe
X, y = data.iloc[:,1:data.shape[1]], data.loc[:, 'Group']

# Leave one out cross validation

loo = LeaveOneOut()

loo_lr = []
loo_lda = []
loo_svm = []
loo_lr_2 = []
loo_lda_2 = []
loo_svm_2 = []
split = loo.split(X)


for train_index, test_index in split:
    # Get training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Logistic regression
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    loo_lr_2.append([clf.score(X_test, y_test), str(y_test.index)])
    loo_lr.append(clf.score(X_test, y_test))
    
    # Linear discriminant analysis
    clf_2 = LinearDiscriminantAnalysis(solver='lsqr').fit(X_train, y_train)
    loo_lda_2.append([clf_2.score(X_test, y_test), str(y_test.index)])
    loo_lda.append(clf_2.score(X_test, y_test))
    
    # Support vector machine
    clf_3 = svm.SVC()
    clf_3.fit(X_train, y_train)
    loo_svm_2.append([clf_3.score(X_test, y_test), str(y_test.index)])
    loo_svm.append(clf_3.score(X_test, y_test))

# Calculate scores for each model
if len(loo_lr) != 0:    
    loo_lr_score = sum(loo_lr)/len(loo_lr)
if len(loo_lda) !=0:
    loo_lda_score = sum(loo_lda)/len(loo_lda)
if len(loo_svm) !=0:
    loo_svm_score = sum(loo_svm)/len(loo_svm)
    
    
###lda_score = 0.50704, lr_score = 0.563380, svm_score = 0.5493