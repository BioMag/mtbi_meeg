#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:14:06 2022

@author: heikkiv

Does (spaghetti) plots of the log-PSDs. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from readdata import chosen_tasks, dataframe as df

#check https://www.python-graph-gallery.com/123-highlight-a-line-in-line-plot for deviations.


#TODO: vectorized data back to matrix (n*m), from which we should calculate global powers?
#So each df row now has [ch1_freq1, ..., ch64_freq89, ch2_freq1, ..., ch64_freq1, ...ch64_freq64] 

#-> revert these? or take from before? 

# Change the style of plot
plt.style.use('seaborn-darkgrid')

plt.figure()