#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:48:59 2022

@author: heikkiv

Script that applies maxfiltering on the bidsified data.
Does all tasks in a row.
"""

import subprocess
import argparse
import os
import mne
from config_eeg import get_all_fnames, fname

#TODO: maxfilter each task, save to bidsified. IN PROGRESS
mne.set_log_level('INFO')

#Handle commandline arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

all_fnames = zip(
    get_all_fnames(args.subject, kind='raw'), #raw data
    get_all_fnames(args.subject, kind='tsss'), #maxfiltered data
    get_all_fnames(args.subject, kind='pos'), #position file
    get_all_fnames(args.subject, kind='tsss_log'), #log files
)


for input_f, output_f, pos_f, log_f in all_fnames:
    
    # arguments given to maxfilter program. TODO: check these!
    args = ['/neuro/bin/util/maxfilter', '-f', input_f, '-o', output_f, '-trans', \
                 input_f, '-st', '16', \
                 '-v','-autobad','on','-origin','fit',\
                 '-in', '8', '-out', '3', '-frame','head', \
                  '-hp', pos_f ]  
        
    
    #save the error log
    log_output = open(log_f, "w")
    
    # run maxfilter, FIXME
    subprocess.run(args=args, stdout=log_output,stderr=log_output)