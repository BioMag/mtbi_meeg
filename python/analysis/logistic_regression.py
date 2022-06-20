#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:01:16 2022

@author: aino
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from readdata import data_frame
import pandas as pd

X, y = data_frame.iloc[:,1:data_frame.shape[1]], data_frame.loc[:, 'Group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
skf = StratifiedKFold(n_splits=4)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

print(clf.score(X_test, y_test))

clf_2 = LinearDiscriminantAnalysis().fit(X_train, y_train)

print(clf_2.score(X_test, y_test))

clf_3 = svm.SVC()

clf_3.fit(X_train, y_train)

print(clf_3.score(X_test, y_test))

