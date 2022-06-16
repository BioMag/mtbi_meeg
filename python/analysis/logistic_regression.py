#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:01:16 2022

@author: aino
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from readdata import data_frame
import pandas as pd

X, y = data_frame.iloc[:,1:data_frame.shape[1]], data_frame.loc[:, 'Group']

clf = LogisticRegression(random_state=0).fit(X, y)

clf.score(X, y)
