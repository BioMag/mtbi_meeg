#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:05:01 2021

@author: kaltiah1
"""

import mne
import os.path as op

file_path = '/projects/tbi_meg/rawdata/'
subject = 'case_5972'
session = '210318'
task = 'eo_ec.fif'

file_name = op.join(file_path,subject,session,task)

raw = mne.io.read_raw_fif(file_name, preload=True)
raw.filter(0,40)

raw.plot(duration=10, n_channels=35)

