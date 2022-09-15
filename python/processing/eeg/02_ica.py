"""
Remove EOG artifacts through independant component analysis (ICA).
No ECG artifacts are removed at the moment, since they are hard to detect based
on just EEG data.


Running: 
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)
"""

import argparse
from collections import defaultdict

from mne import Epochs
from mne.io import read_raw_fif
from mne.preprocessing import find_eog_events, find_ecg_events, ICA
from mne import open_report
import datetime

from config_eeg import get_all_fnames, task_from_fname, fname, ecg_channel

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
    get_all_fnames(args.subject, kind='filt_meg', exclude=exclude),
    get_all_fnames(args.subject, kind='ica_meg', exclude=exclude),
    get_all_fnames(args.subject, kind='clean_meg', exclude=exclude),
)

for filt_fname, ica_fname, clean_fname in all_fnames:
    task = task_from_fname(filt_fname)
    
    if 'PASAT' in task:
    #TODO: crop first and last 2-5 s
        raw_filt = read_raw_fif(filt_fname, preload=True)
    
        # Run a detection algorithm for the onsets of eye blinks (EOG) and heartbeat artefacts (ECG)
        eog_events = find_eog_events(raw_filt)
        eog_epochs = Epochs(raw_filt, eog_events, tmin=-0.5, tmax=0.5, preload=True)
       
        try:
            ecg_events, ch, _ = find_ecg_events(raw_filt, ch_name=ecg_channel)
        except ValueError:
            ecg_events, ch, _ =  find_ecg_events(raw_filt)
        ecg_epochs = Epochs(raw_filt, ecg_events, tmin=-0.5, tmax= 0.5, preload=True)

        # Perform ICA decomposition
        ica = ICA(n_components=0.99, random_state=0).fit(raw_filt)
    
        # Find components that are likely capturing EOG artifacts
        
        bads_eog, scores_eog = ica.find_bads_eog(eog_epochs, threshold=3.0)
        print('Bads EOG:', bads_eog)
        try:
            bads_ecg, scores_ecg = ica.find_bads_ecg(raw_filt, method='correlation', threshold=2.0)
        except ValueError:
            print('Not able to find ecg components')
            bads_ecg = []
            scores_ecg = []
            # # ECG artefact removal for subjects who don't have ECG data
            # bads_ecg, scores_ecg = ica.find_bads_ecg(raw_filt, method = 'correlation', threshold='auto')
            # # ValueError: Unable to generate artificial ECG channel - is this even possible at all for EEG?
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
                                eog_evoked=eog_epochs.average(),
                                eog_scores=scores_eog,
                                tags=(f'{task}', 'EOG', 'ICA'),
                                replace=True
                                )
  
            if len(bads_ecg)>0:
                report.add_ica(ica=ica, 
                                    title=f' {task}' + ' ECG', 
                                    inst=raw_filt, 
                                    picks=bads_ecg,
                                    ecg_evoked=ecg_epochs.average(),
                                    ecg_scores=scores_ecg,
                                    tags=(f'{task}', 'ECG', 'ICA'),
                                    replace=True
                                )
            # report.add_figure(
            #     ica.plot_scores(scores_eog, exclude=bads_eog, title=date_time, show=False),
            #     f'{task}: EOG scores', replace=True, tags=('EOG', f'{task}', 'ICA'))
    
            report.add_figure(
                ica.plot_overlay(eog_epochs.average(), title=date_time, show=False),
                f'{task}: EOG overlay', replace=True, tags=(f'{task}', 'ICA', 'EOG', 'overlay'))
            
  
            #     report.add_figure(
            #         ica.plot_scores(scores_ecg, exclude=bads_ecg, title=date_time, show=False),
            #         f'{task}: ECG scores', replace=True, tags=('ECG', f'{task}', 'ICA'))
                
            report.add_figure(
                 ica.plot_overlay(ecg_epochs.average(), title=date_time, show=False),
                 f'{task}: ECG overlay', replace=True, tags=(f'{task}', 'ICA', 'ECG', 'overlay'))
                

    
            report.save(fname.report_html(subject=args.subject),
                        overwrite=True, open_browser=False)
        
