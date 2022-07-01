#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:45:44 2019

Simple example for how to use mne_bids to save your data in BIDS format

Change according to your needs

@author: mhusberg
"""

import os.path as op
import mne

from mne_bids import (write_raw_bids, BIDSPath, print_dir_tree)

# Test on the sample dataset, preprocessed with the download_example_data to create extra 'runs' and 'sessions'
# from mne.datasets import sample
# data_path = sample.data_path()

# Your own path
#data_path = '/m/myproject/'
data_path = '/projects/tbi_meg/'
data_folder = 'k22_eo_ec_for_BIDS'

# Output file
output_path = op.join(data_path, 'BIDS')

# Define event IDs according to your project
#events = {'eyes_closed.fif':1}#, 'eyes_open.fif"': 2, 'PASAT_1raw.fif': 3, 'PASAT_2raw.fif': 4}

# For the sample data:
#events = {"eyes_closed.fif":ec, "eyes_open.fif":eo, "PASAT_1raw":pasat1, "PASAT_2raw":pasat2} 

def create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task): # add date if necessary event id omitted

    # Read in data
    #raw_fname = op.join(data_path, subject_in, session_in, filename) # Folder structure for recorded raw data
    raw_fname = op.join(data_path, data_folder, subject_in, session_in, filename) # Folder structure for recorded raw data
    #raw_fname = op.join(data_path, 'MEG', subject_in, date, filename) # Add date if necessary
    raw = mne.io.read_raw_fif(raw_fname)
   # events_data = mne.find_events(raw, min_duration=2/1000.)

    bids_path = BIDSPath(subject=subject_out, 
                         session=session, 
                         task=task, 
                         run=run, 
                         processing=proc, 
                         root=output_path)
    
    write_raw_bids(raw, 
                   bids_path, 
                   #events_data=events_data,
                   overwrite=True)

    print_dir_tree(output_path)


#######################################################################


# Sample subject
subject_in = 'case_5972'
subject_out = '19C' # subject-id: '01' -> 'sub-01' in BIDS structure

# Session 1
#date = 'YYMMDD' # date when measured, if in folder structure
session = '01'
session_in = '210318'
proc = 'raw' 

#####
# Task1
task = 'ec'
# Filename, run 1
filename='eyes_closed.fif' # Fill in name of rawdata-file 1
run = '01'

create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary, task omitted, events omitted

# Task2
task = 'eo'
# Filename, run 1
filename='eyes_open.fif' # Fill in name of rawdata-file 2
run = '01'

create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

# # Task3
# task = 'PASAT'
# # Filename, run 1
# filename='PASAT_1_raw.fif' # Fill in name of rawdata-file 2
# run = '01'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

# filename='PASAT_2_raw.fif' # Fill in name of rawdata-file 2
# run = '02'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

# #######################################################################
# # Sample subject
# subject_in = 'case_6050'
# subject_out = '30P' # subject-id: '01' -> 'sub-01' in BIDS structure

# # Session 2
# #date = 'YYMMDD' # date when measured, if in folder structure
# session = '02'
# session_in = '210504'
# proc = 'raw' 

# ##########
# # Task1
# task = 'ec'
# # Filename, run 1
# filename='eyes_closed.fif' # Fill in name of rawdata-file 1
# run = '01'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary, task omitted, events omitted

# # Task2
# task = 'eo'
# # Filename, run 1
# filename='eyes_open.fif' # Fill in name of rawdata-file 2
# run = '01'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

# # Task3
# task = 'PASAT'
# # Filename, run 1
# filename='pasat1raw.fif' # Fill in name of rawdata-file 2
# run = '01'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

# filename='pasat2raw.fif' # Fill in name of rawdata-file 2
# run = '02'

# create_BIDS(filename, subject_in, subject_out, session, session_in, run, proc, task) # add date if necessary

