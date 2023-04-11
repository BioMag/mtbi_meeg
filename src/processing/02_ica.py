"""
Remove EOG & ECG artifacts through independant component analysis (ICA).


Running: 
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)
"""

import argparse
from collections import defaultdict

from mne import Epochs, set_log_level
from mne.io import read_raw_fif
from mne.preprocessing import create_eog_epochs, create_ecg_epochs, ICA
from mne import open_report
import datetime
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config_eeg import get_all_fnames, task_from_fname, fname, ecg_channel


# Save time of beginning of the execution to measure running time
start_time = time.time()

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom'] #these don't have eye blinks.
bad_subjects = ['01P', '02P', '03P', '04P', '05P', '06P', '07P']#these ica need to be done manually
all_fnames = zip(
    get_all_fnames(args.subject, kind='filt', exclude=exclude),
    get_all_fnames(args.subject, kind='ica', exclude=exclude),
    get_all_fnames(args.subject, kind='clean', exclude=exclude),
)

for filt_fname, ica_fname, clean_fname in all_fnames:
    task = task_from_fname(filt_fname)
    #TODO: crop first and last 2-5 s
    raw_filt = read_raw_fif(filt_fname, preload=True)
    
    # Reduce logging level (technically, one could define it in the read_raw_fif function, but it seems to be buggy)
    # More info about the bug can be found here: https://github.com/mne-tools/mne-python/issues/8872
    set_log_level(verbose='Warning')

    # Run a detection algorithm for the onsets of eye blinks (EOG) and heartbeat artefacts (ECG)
    eog_events = create_eog_epochs(raw_filt)
    #TODO: skip eog events for ec
    if ecg_channel in raw_filt.info['ch_names']:
                        ecg_events = create_ecg_epochs(raw_filt)
                        ecg_exists = True
    else:
        ecg_exists = False
    # Perform ICA decomposition
    ica = ICA(n_components=0.99, random_state=0).fit(raw_filt)

    # Find components that are likely capturing EOG artifacts
    
    bads_eog, scores_eog = ica.find_bads_eog(raw_filt)
    print('Bads EOG:', bads_eog)
    try:
        bads_ecg, scores_ecg = ica.find_bads_ecg(raw_filt, method='correlation',threshold='auto')
    except ValueError:
        print('Not able to find ecg components')
        bads_ecg = []
        scores_ecg = []
    # Mark the EOG components for removal
    ica.exclude = bads_eog + bads_ecg
    ica.save(ica_fname, overwrite=True) 

    # Remove the EOG artifact components from the signal.
    raw_ica = ica.apply(raw_filt)
    raw_ica.save(clean_fname, overwrite=True)

    # Date and time
    now = datetime.datetime.now()
    date_time = now.strftime('%A, %d. %B %Y %I:%M%p')

    # Put a whole lot of quality control figures in the HTML report.
    with open_report(fname.report(subject=args.subject)) as report:
        if len(bads_eog)>0:
            report.add_ica(ica=ica, 
                            title=f' {task}' + ' EOG', 
                            inst=raw_filt, 
                            picks=bads_eog,
                            eog_evoked=eog_events.average(),
                            eog_scores=scores_eog,
                            tags=(f'{task}', 'EOG', 'ICA'),
                            replace=True
                            )
        if ecg_exists:
            if len(bads_ecg)>0:
                report.add_ica(ica=ica, 
                                title=f' {task}' + ' ECG', 
                                inst=raw_filt, 
                                picks=bads_ecg,
                                ecg_evoked=ecg_events.average(),
                                ecg_scores=scores_ecg,
                                tags=(f'{task}', 'ECG', 'ICA'),
                                replace=True
                                )

        report.add_figure(
            ica.plot_overlay(eog_events.average(), title=date_time, show=False),
            f'{task}: EOG overlay', replace=True, tags=(f'{task}', 'ICA', 'EOG', 'overlay'))
        
        if ecg_exists:

            
            report.add_figure(
                ica.plot_overlay(ecg_events.average(), title=date_time, show=False),
                f'{task}: ECG overlay', replace=True, tags=(f'{task}', 'ICA', 'ECG', 'overlay'))
            

        report.save(fname.report_html(subject=args.subject),
                    overwrite=True, open_browser=False)
    
    with open("ecg_missing.txt", "a") as file:
        file_name = task_from_fname(filt_fname)
        if not ecg_exists:
            print(f'{args.subject}: no ECG found') 
            file.write(str(args.subject)+file_name+'\n')
        file.close()

# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time of 02_ica.py is: {round(execution_time,2)} seconds\n')
print('###################################################\n')
