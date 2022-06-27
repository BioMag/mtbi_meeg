"""
Perform bandpass filtering and notch filtering to get rid of cHPI and powerline
frequencies.



Running:
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)
"""

import argparse
from collections import defaultdict

from mne.io import read_raw_fif
from mne import open_report

from config_eeg import get_all_fnames, fname, ec_bads, eo_bads, pasat1_bads, pasat2_bads, fmin, fmax, fnotch

#TODO: fix 35C channels

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom'] #these don't have eye blinks.
# bad_subjects = ['01P', '02P', '03P', '04P', '05P', '06P', '07P']#these ica need to be done manually
all_fnames = zip(
    get_all_fnames(args.subject, kind='raw', exclude=exclude),
    get_all_fnames(args.subject, kind='filt', exclude=exclude),
)
# raw_fnames = get_all_fnames(args.subject, kind='raw')
# filt_fnames = get_all_fnames(args.subject, kind='filt')

for raw_fname, filt_fname in all_fnames:
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
    
    # Remove MEG channels. This is the EEG pipeline after all.
    raw.pick_types(meg=False, eeg=True, eog=True, stim=True, ecg=True, exclude=[])
    
    # Plot segment of raw data
    figures['raw_segment'].append(raw.plot(n_channels=30, show=False))
    
    # Interpolate bad channels
    raw.interpolate_bads()
    figures['interpolated_segment'].append(raw.plot(n_channels=30, show=False))
    
    # Add a plot of the power spectrum to the list of figures to be placed in
    # the HTML report.
    raw_plot = raw.plot_psd(fmin=0, fmax=40, show=False)
    figures['before_filt'].append(raw_plot)

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
    figures['after_filt'].append(filt_plot)
    
    raw.close()

# Write HTML report with the quality control figures
with open_report(fname.report(subject=args.subject)) as report:
    report.add_slider_to_section(
        figures['before_filt'],
        section='freq filter',
        title='Before frequency filtering',
        replace=True,
    )
    report.add_slider_to_section(
        figures['after_filt'],
        section='freq filter',
        title='After frequency filtering',
        replace=True,
    )
    report.add_slider_to_section(
        figures['raw_segment'],
        section='freq filter',
        title='Before interpolation',
        replace=True,
    )
    report.add_slider_to_section(
        figures['interpolated_segment'],
        section='freq filter',
        title='After interpolation',
        replace=True,
    )
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)
