"""
Perform bandpass filtering and notch filtering to get rid of cHPI and powerline
frequencies.
"""
import argparse
from collections import defaultdict

from mne.io import read_raw_fif
from mne import open_report

from config_eeg import get_all_fnames, fname, bads, fmin, fmax, fnotch

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', type=int, help='The subject to process')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
raw_fnames = get_all_fnames(args.subject, kind='raw')
filt_fnames = get_all_fnames(args.subject, kind='filt')

for raw_fname, filt_fname in zip(raw_fnames, filt_fnames):
    raw = read_raw_fif(raw_fname, preload=True)

    # Remove MEG channels. This is the EEG pipeline after all.
    raw.pick_types(meg=False, eeg=True, eog=True, stim=True)

    # Mark bad channels that were manually annotated earlier.
    raw.info['bads'] = bads[args.subject]

    # Add a plot of the power spectrum to the list of figures to be placed in
    # the HTML report.
    figures['before_filt'].append(raw.plot_psd())

    # Remove 50Hz power line noise (and the first harmonic: 100Hz)
    filt = raw.notch_filter(fnotch, picks=['eeg', 'eog'])

    # Apply bandpass filter
    filt = raw.filter(fmin, fmax, picks=['eeg', 'eog'])

    # Save the filtered data
    filt_fname.parent.mkdir(parents=True, exist_ok=True)
    filt.save(filt_fname, overwrite=True)

    # Add a plot of the power spectrum of the filtered data to the list of
    # figures to be placed in the HTML report.
    figures['after_filt'].append(filt.plot_psd())

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
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)
