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

from config_eeg import get_all_fnames, task_from_fname, fname, ecg_channel

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom', 'ec', 'eo'] #these don't have eye blinks.
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

    # Run a detection algorithm for the onsets of eye blinks (EOG) and heartbeat artefacts (ECG)
    eog_events = find_eog_events(raw_filt)
    eog_epochs = Epochs(raw_filt, eog_events, tmin=-0.5, tmax=0.5, preload=True)
    #TODO: skip eog events for ec
    if ecg_channel in raw_filt.info['ch_names']:
                        ecg_events, ch, _ = find_ecg_events(raw_filt, ch_name=ecg_channel)
                        ecg_epochs = Epochs(raw_filt, ecg_events, tmin=-0.5, tmax= 0.5, preload=True)
                        ecg_exists = True
    else:
        ecg_exists = False
    # Perform ICA decomposition
    ica = ICA(n_components=0.99, random_state=0).fit(raw_filt)

    # Find components that are likely capturing EOG artifacts
    
    bads_eog, scores_eog = ica.find_bads_eog(eog_epochs)
    print('Bads EOG:', bads_eog)
    try:
        bads_ecg, scores_ecg = ica.find_bads_ecg(raw_filt, method='ctps', threshold='auto')
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

    # Put a whole lot of quality control figures in the HTML report.
    with open_report(fname.report(subject=args.subject)) as report:
        report.add_figs_to_section(
            ica.plot_scores(scores_eog, exclude=bads_eog, show=False),
            f'{task}: EOG scores', section='ICA', replace=True)

        report.add_figs_to_section(
            ica.plot_overlay(eog_epochs.average(), show=False),
            f'{task}: EOG overlay', section='ICA', replace=True)
        
        if ecg_exists:
            report.add_figs_to_section(
                ica.plot_scores(scores_ecg, exclude=bads_ecg, show=False),
                f'{task}: ECG scores', section='ICA', replace=True)
            
            report.add_figs_to_section(
                ica.plot_overlay(ecg_epochs.average(), show=False),
                f'{task}: ECG overlay', section='ICA', replace=True)

        if len(bads_eog) == 1:
            report.add_figs_to_section(
                ica.plot_properties(eog_epochs, bads_eog, show=False),
                [f'{task}: Component {i:02d}' for i in bads_eog],
                section='ICA', replace=True)
        elif len(bads_eog) > 1:
            report.add_slider_to_section(
                ica.plot_properties(eog_epochs, bads_eog, show=False),
                captions=[f'{task}: Component {i:02d}' for i in bads_eog],
                title=f'{task}: EOG component properties', section='ICA',
                replace=True)

        report.save(fname.report_html(subject=args.subject),
                    overwrite=True, open_browser=False)
    
    with open("ecg_puuttuu.txt", "a") as filu:
        filu_name = task_from_fname(filt_fname)
        if not ecg_exists:
            filu.write(str(args.subject)+filu_name+'\n')
        filu.close()
