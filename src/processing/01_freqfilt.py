"""
Perform bandpass filtering and notch filtering to get rid of cHPI and powerline
frequencies.


Running:
import subprocess
subprocess.run('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/processing/eeg/runsome.sh', shell=True)
"""

import argparse
from collections import defaultdict
from mne.io import read_raw_fif
from mne import open_report, set_log_level
import datetime
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Save time of beginning of the execution to measure running time
start_time = time.time()

from config_common import processed_data_dir
from config_eeg import get_all_fnames, fname, ec_bads, eo_bads, pasat1_bads, pasat2_bads, fmin, fmax, fnotch

#TODO: fix 35C channels (WHAT'S THIS???)

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process', default='10C')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom'] #these don't have eye blinks.
all_fnames = zip(
    get_all_fnames(args.subject, kind='raw', exclude=exclude),
    get_all_fnames(args.subject, kind='filt', exclude=exclude),
)


# Date and time
now = datetime.datetime.now()
date_time = now.strftime('%A, %d. %B %Y %I:%M%p')

corrupted_raw_files = []

for raw_fname, filt_fname in all_fnames:
    try:
        raw = read_raw_fif(raw_fname, preload=True)

        # Reduce logging level (technically, one could define it in the read_raw_fif function, but it seems to be buggy)
        # More info about the bug can be found here: https://github.com/mne-tools/mne-python/issues/8872
        set_log_level(verbose='Warning')

        # Mark bad channels that were manually annotated earlier.
        raw_str = str(raw_fname)
        if 'task-ec' in raw_str:
            raw.info['bads'] = ec_bads[args.subject]
            task = 'ec'
        elif 'task-eo' in raw_str:
            raw.info['bads'] = eo_bads[args.subject]
            task = 'eo'
        elif 'task-PASAT' in raw_str and 'run-01' in raw_str:
            raw.info['bads'] = pasat1_bads[args.subject]
            task = 'pasat1'
        elif 'task-PASAT' in raw_str and 'run-02' in raw_str:
            raw.info['bads'] = pasat2_bads[args.subject]
            task = 'pasat2'
        
        # Remove MEG channels. This is the EEG pipeline after all.
        raw.pick_types(meg=False, eeg=True, eog=True, stim=True, ecg=True, exclude=[])
        
        # Plot segment of raw data
        figures['raw segment'].append(raw.plot(n_channels=30, title = date_time, show=False))
        
        # Interpolate bad channels
        raw.interpolate_bads()
        figures['interpolated segment'].append(raw.plot(n_channels=30, title = date_time + task, show=False))
        
        # Add a plot of the power spectrum to the list of figures to be placed in
        # the HTML report.
        raw_plot = raw.compute_psd(fmin=0, fmax=40).plot(show=False)
        figures['before filt'].append(raw_plot)
    
        # Remove 50Hz power line noise (and the first harmonic: 100Hz)
        filt = raw.notch_filter(fnotch, picks=['eeg', 'eog', 'ecg'])
        
        # Apply bandpass filter
        filt = filt.filter(fmin, fmax, picks=['eeg', 'eog', 'ecg'])
    
        # Save the filtered data
        filt_fname.parent.mkdir(parents=True, exist_ok=True)
        filt.save(filt_fname, overwrite=True)
    
        # Add a plot of the power spectrum of the filtered data to the list of
        # figures to be placed in the HTML report.
        filt_plot = filt.plot_psd(fmin=0, fmax=40, show=False)
        figures['after filt'].append(filt_plot)
        
        raw.close()
    except:
        corrupted_raw_files.append(args.subject)
    


# Write HTML report with the quality control figures
# TODO: These could be nicer!
section='Filtering'
with open_report(fname.report(subject=args.subject)) as report:
    report.add_figure(
        figures['before filt'],
        title='Before frequency filtering',
        caption=('Eyes open', 'Eyes closed', 'PASAT run 1', 'PASAT run 2'),
        replace=True,
        section=section,
        tags=('filt')
    )
    report.add_figure(
        figures['after filt'],
        title='After frequency filtering',
        caption=('Eyes open', 'Eyes closed', 'PASAT run 1', 'PASAT run 2'),
        replace=True,
        section=section,
        tags=('filt')
    )
    report.add_figure(
        figures['raw segment'],
        title='Before interpolation',
        caption=('Eyes open', 'Eyes closed', 'PASAT run 1', 'PASAT run 2'),
        replace=True,
        section=section,
        tags=('raw')
    )
    report.add_figure(
        figures['interpolated segment'],
        title='After interpolation',
        caption=('Eyes open', 'Eyes closed', 'PASAT run 1', 'PASAT run 2'),
        replace=True,
        section=section,
        tags=('raw')
    )
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)

# TODO: Once the folder structure is defined, re-code the path depending on where is this expected
# with open('maxfilter_corrupted_or_missing.txt', 'a') as  
with open('/net/tera2/home/portae1/biomag/mtbi-eeg/python/processing/eeg/maxfilter_puuttuu.txt', 'a') as file:
    for bad_file in corrupted_raw_files:
        file.write(bad_file+'\n')
    file.close()

# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time of 01_freqfilter.py is: {round(execution_time,2)} seconds\n')
print('###################################################\n')
