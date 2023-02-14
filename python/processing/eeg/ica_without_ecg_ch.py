#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for handling the ICA for subjects without ECG sensor. 
Builds virtual ECG chennel from magnetometers.

Created on Thu Jun 30 14:12:52 2022

@author: aino

Running: 
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runsome.sh', shell=True)
"""
import argparse
from collections import defaultdict

from mne import Epochs
from mne.io import read_raw_fif
from mne.preprocessing import find_eog_events, find_ecg_events, ICA
from mne import open_report
import datetime

from config_eeg import get_all_fnames, task_from_fname, fname, ecg_channel, ec_bads, eo_bads, pasat1_bads, pasat2_bads


# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# For collecting figures for quality control
figures = defaultdict(list)

exclude = []

all_fnames = zip(get_all_fnames(args.subject, kind='filt', exclude=exclude),
                 get_all_fnames(args.subject, kind='ica', exclude=exclude),
                 get_all_fnames(args.subject, kind='tsss', exclude=exclude),
                 get_all_fnames(args.subject, kind='clean', exclude=exclude))

for filt_fname, ica_fname, raw_fname, clean_fname in all_fnames:
    task = task_from_fname(filt_fname)
    raw = read_raw_fif(raw_fname, preload=True)
    
        # Mark bad channels that were manually annotated earlier.
    raw_str = str(raw_fname)
    if 'task-ec' in raw_str:
        raw.info['bads'] = ec_bads[args.subject]
    elif 'task-eo' in raw_str:
        raw.info['bads'] = eo_bads[args.subject]
    elif 'task-PASAT' in raw_str and 'run-01' in raw_str:
        raw.info['bads'] = pasat1_bads[args.subject]
    elif 'task-PASAT' in raw_str and 'run-02' in raw_str:
        raw.info['bads'] = pasat2_bads[args.subject]
    
    # Date and time
    now = datetime.datetime.now()
    date_time = now.strftime('%A, %d. %B %Y %I:%M%p')
    
    # Plot segment of raw data
    figures['raw_segment'].append(raw.plot(n_channels=30, title = date_time, show=False))

    
    # Interpolate bad channels
    raw.interpolate_bads()
    figures['interpolated_segment'].append(raw.plot(n_channels=30, title = date_time, show=False))
    
    # Remove 50Hz power line noise (and the first harmonic: 100Hz)
    filt = raw.notch_filter(50, picks=['eeg', 'eog', 'ecg', 'meg'])
    
    # Apply bandpass filter
    filt = raw.filter(1, 90, picks=['eeg','eog','ecg','meg'])
    
    # Run a detection algorithm for the onsets of eye blinks (EOG) and heartbeat artefacts (ECG)
    eog_events = find_eog_events(filt)
    eog_epochs = Epochs(filt, eog_events, tmin=-0.5, tmax=0.5, preload=True)
    
    ecg_events, ch, _ = find_ecg_events(filt)
    ecg_epochs = Epochs(filt, ecg_events, tmin = -0.5, tmax = 0.5, preload=True)
    
    # Perform ICA decomposition
    ica = ICA(n_components=0.99, random_state=0).fit(filt)
    
    # Find components that are likely capturing EOG artifacts
    bads_eog, scores_eog = ica.find_bads_eog(eog_epochs, threshold=2.0)
    # Find components that are likely capturing ECG artifacts
    bads_ecg, scores_ecg = ica.find_bads_ecg(filt, method = 'correlation', threshold='auto')
    #TODO: ecg_epochs????
    # Remove MEG channels 
    raw.pick_types(meg=False, eeg=True, eog=False, stim=False, ecg=False, exclude=[])
    
    # Mark the components for removal
    ica.exclude = bads_eog + bads_ecg
    ica.save(ica_fname, overwrite=True)
    
    # Date and time
    now = datetime.datetime.now()
    date_time = now.strftime('%A, %d. %B %Y %I:%M%p')

    # Put a whole lot of quality control figures in the HTML report.
    with open_report(fname.report(subject=args.subject)) as report:
         # if len(bads_eog)>0:
         #     report.add_ica(ica=ica, 
         #                     title=f' {task}' + ' EOG', 
         #                     inst=filt, 
         #                     picks=bads_eog,
         #                     eog_evoked=eog_epochs.average(),
         #                     eog_scores=scores_eog,
         #                     tags=(f'{task}', 'EOG', 'ICA'),
         #                     replace=True
         #                     )
         # if len(bads_ecg)>0:
         #    report.add_ica(ica=ica, 
         #                    title=f' {task}' + ' ECG', 
         #                    inst=filt, 
         #                    picks=bads_ecg,
         #                    ecg_evoked=ecg_epochs.average(),
         #                    ecg_scores=scores_ecg,
         #                    tags=(f'{task}', 'ECG', 'ICA'),
         #                    replace=True
         #                    )
          report.add_figure(
              ica.plot_scores(scores_eog, exclude=bads_eog, title=date_time, show=False),
              f'{task}: EOG scores', replace=True, tags=('EOG', f'{task}', 'ICA'))

          report.add_figure(
              ica.plot_overlay(eog_epochs.average(), title=date_time, show=False),
              f'{task}: EOG overlay', replace=True, tags=(f'{task}', 'ICA', 'EOG', 'overlay'))
         

          report.add_figure(
              ica.plot_scores(scores_ecg, exclude=bads_ecg, title=date_time, show=False),
              f'{task}: ECG scores', replace=True, tags=('ECG', f'{task}', 'ICA'))
             
          report.add_figure(
              ica.plot_overlay(ecg_epochs.average(), title=date_time, show=False),
              f'{task}: ECG overlay', replace=True, tags=(f'{task}', 'ICA', 'ECG', 'overlay'))
             
          if len(bads_ecg) == 1:
             report.add_figure(
                 ica.plot_properties(ecg_epochs, bads_ecg, show=False),
                 [f'{task}: ECG Component {i:02d}' for i in bads_ecg],
                 replace=True, tags=('ECG', f'{task}', 'ICA'))
          elif len(bads_ecg) > 1:
              report.add_figure(
                ica.plot_properties(ecg_epochs, bads_ecg, show=False),
                f'{task}: ECG component properties', 
                #captions=[f'{task}: Component {i:02d}' for i in bads_ecg],
                replace=True, tags=('ECG', f'{task}', 'ICA'))
                 
                 
          if len(bads_eog) == 1:
              report.add_figure(
                  ica.plot_properties(eog_epochs, bads_eog, show=False),
                  [f'{task}: EOG Component {i:02d}' for i in bads_eog],
                  replace=True, tags=('EOG', f'{task}', 'ICA'))
          elif len(bads_eog) > 1:
              report.add_figure(
                  ica.plot_properties(eog_epochs, bads_eog, show=False),
                  f'{task}: EOG component properties',
                  #captions=[f'{task}: Component {i:02d}' for i in bads_eog],
                  replace=True, tags=('EOG', f'{task}', 'ICA'))

          report.save(fname.report_html(subject=args.subject),
                    overwrite=True, open_browser=False)
    
    
    
    
    
    
