#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:01:16 2022

@author: aino

Tests the performance of three models (logistic regression, linear discriminant analysis 
and support vector machine) using stratified k fold cross validation.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
import numpy as np
from readdata import data_frame
import pandas as pd

X, y = data_frame.iloc[:,1:data_frame.shape[1]], data_frame.loc[:, 'Group']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

# Stratified K fold cross validation
stratified_accuracy_lr =[]
# stratified_accuracy_lda =[]
# stratified_accuracy_svm = []
skf = StratifiedKFold(n_splits=7)
split = skf.split(X, y)

# pca = PCA(n_components=4)

# for train_index, test_index in split:
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     # pca_x_train = pca.fit_transform(X_train)
#     # pca_x_test = pca.fit_transform(X_test)
#     # clf_pca = LogisticRegression().fit(pca_x_train, y_train)
#     # pca_score = clf_pca.score(pca_x_test, y_test)
#     # stratified_accuracy_lr.append(pca_score)
#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)
#     stratified_accuracy_lr.append(clf.score(X_test, y_test))
# #     clf_2 = LinearDiscriminantAnalysis(solver='lsqr').fit(X_train, y_train)
# #     stratified_accuracy_lda.append(clf_2.score(X_test, y_test))
# #     clf_3 = svm.SVC()
# #     clf_3.fit(X_train, y_train)
# #     stratified_accuracy_svm.append(clf_3.score(X_test, y_test))
    
clf = LogisticRegression(random_state=0).fit(X_train, y_train)  
print(clf.score(X_test, y_test))

# clf_2 = LinearDiscriminantAnalysis().fit(X_train, y_train)
# print(clf_2.score(X_test, y_test))

# clf_3 = svm.SVC()
# clf_3.fit(X_train, y_train)
# print(clf_3.score(X_test, y_test))

# # Principal component analysis
# pca = PCA(n_components=4)
# pca_x_train = pca.fit_transform(X_train)
# pca_x_test = pca.fit_transform(X_test)

# clf_pca = LogisticRegression().fit(pca_x_train, y_train)
# pca_score = clf_pca.score(pca_x_test, y_test)



