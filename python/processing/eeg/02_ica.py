"""
Remove EOG artifacts through independant component analysis (ICA).
No ECG artifacts are removed at the moment, since they are hard to detect based
on just EEG data.
"""
import argparse
from collections import defaultdict

from mne import Epochs
from mne.io import read_raw_fif
from mne.preprocessing import find_eog_events, ICA
from mne import open_report

from config_eeg import get_all_fnames, task_from_fname, fname

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', type=int, help='The subject to process')
args = parser.parse_args()

# Along the way, we collect figures for quality control
figures = defaultdict(list)

# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom', 'eyesclosed']  # These don't have eye blinks.
all_fnames = zip(
    get_all_fnames(args.subject, kind='filt', exclude=exclude),
    get_all_fnames(args.subject, kind='ica', exclude=exclude),
    get_all_fnames(args.subject, kind='clean', exclude=exclude),
)

for filt_fname, ica_fname, clean_fname in all_fnames:
    task = task_from_fname(filt_fname)

    raw_filt = read_raw_fif(filt_fname, preload=True)

    # Run a detection algorithm for the onsets of eye blinks (EOG).
    eog_events = find_eog_events(raw_filt)

    # Perform ICA decomposition
    ica = ICA(random_state=0).fit(raw_filt)

    # Find components that are likely capturing EOG artifacts
    eog_epochs = Epochs(raw_filt, eog_events, tmin=-0.5, tmax=0.5,
                        preload=True)
    bads_eog, scores_eog = ica.find_bads_eog(eog_epochs)
    print('Bads EOG:', bads_eog)

    # Mark the EOG components for removal
    ica.exclude = bads_eog
    ica.save(ica_fname)

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
