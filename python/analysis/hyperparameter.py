#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:49:07 2022

@author: aino
"""

import argparse
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from readdata import dataframe

# Deal with command line arguments

parser = argparse.ArgumentParser()

parser.add_argument('--clf', type=str, help='classifier')
parser.add_argument('--task', type=str, help='task')
parser.add_argument('--parameters', type=dict, help='')
parser.add_argument()
parser.add_argument()

args = parser.parse_args()

# Number of random trials
NUM_TRIALS = 10

# Get data
X, y = dataframe.iloc[:,1:dataframe.shape[1]], dataframe.loc[:, 'Group']

# Set up possible values of parameters to optimize over
param_grid = {'C':[1,10,100], 'penalty': ['l1','l2']}

# Model to optimize
estimator = LogisticRegression(solver='liblinear')

# Array to store scores
nested_scores = np.zeros(NUM_TRIALS)

# Nested cross validation
for i in range(NUM_TRIALS):
    # Choose cross validation methods
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    

    # Nested CV with parameter optimization
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv)
    nested_score = cross_val_score(estimator=estimator, X=X, y=y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
    