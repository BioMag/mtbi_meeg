#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:05:01 2021

@author: kaltiah1
"""

import mne
import os.path as op

file_path = '/projects/tbi_meg/rawdata/'
fileout_path = '/projects/tbi_meg/k22_eo_ec_for_BIDS/'
subject = 'case_5972'
session = '210318'
task = 'eo_ec.fif'
task1 = 'eyes_closed.fif'
task2 = 'eyes_open.fif'
tmin_1, tmax_1 = 0,300
tmin_2, tmax_2 = 305,None

file_name = op.join(file_path,subject,session,task)

#EC_file
raw = mne.io.read_raw_fif(file_name, preload=True)
raw.crop(tmin_1,tmax_1)

fileout_name1 = op.join(fileout_path,subject,session,task1)
raw.save(fileout_name1)

#EO file
raw = mne.io.read_raw_fif(file_name, preload=True)
raw.crop(tmin_2,tmax_2)

fileout_name2 = op.join(fileout_path,subject,session,task2)
raw.save(fileout_name2)