"""
Compute the Power Spectral Density (PSD) for each channel.
"""
import argparse

from mne.io import read_raw_fif
from mne.time_frequency import psd_welch
from mne.externals.h5io import write_hdf5
from mne.viz import iter_topography
from mne import open_report, find_layout, pick_info, pick_types
import matplotlib.pyplot as plt

from config_eeg import fname, n_fft

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', type=int, help='The subject to process')
args = parser.parse_args()

# Compute the PSD for each task
psds = dict()

# Maximum frequency at which to compute PSD
fmax = 40

# Up to now, we have always skipped bad channels. To make sure we can average
# the PSDs later, each recording needs to have the same channels defined. So,
# we call `interpolate_bads()` to replace all channels marked "bads" by an
# interpolation of the surrounding channels. This ensures that each PSD object
# has a signal designed for all channels, hence they are all compatible.
raw = read_raw_fif(fname.filt(subject=args.subject, task='eyesclosed', run=1),
                   preload=True)
raw.interpolate_bads()
psds['eyesclosed'], freqs = psd_welch(raw, fmax=fmax, n_fft=n_fft)

raw = read_raw_fif(fname.clean(subject=args.subject, task='eyesopen', run=1),
                   preload=True)
raw.interpolate_bads()
psds['eyesopen'], freqs = psd_welch(raw, fmax=fmax, n_fft=n_fft)

raw = read_raw_fif(fname.clean(subject=args.subject, task='pasat', run=1),
                   preload=True)
raw.interpolate_bads()
psds['pasat_run1'], freqs = psd_welch(raw, fmax=fmax, n_fft=n_fft)

raw = read_raw_fif(fname.clean(subject=args.subject, task='pasat', run=2),
                   preload=True)
raw.interpolate_bads()
psds['pasat_run2'], freqs = psd_welch(raw, fmax=fmax, n_fft=n_fft)

# Add some metadata to the file we are writing
psds['info'] = raw.info
psds['freqs'] = freqs
write_hdf5(fname.psds(subject=args.subject), psds, overwrite=True)

# Add a PSD plot to the report.
times = [1, 2]
info = pick_info(raw.info, pick_types(raw.info, eeg=True))
layout = find_layout(info)


def on_pick(ax, ch_idx):
    """Create a larger PSD plot for when one of the tiny PSD plots is
       clicked."""
    ax.plot(psds['freqs'], psds['eyesclosed'][ch_idx], color='C0',
            label='eyes closed')
    ax.plot(psds['freqs'], psds['eyesopen'][ch_idx], color='C1',
            label='eyes open')
    ax.plot(psds['freqs'], psds['pasat_run1'][ch_idx], color='C2',
            label='pasat run 1')
    ax.plot(psds['freqs'], psds['pasat_run2'][ch_idx], color='C3',
            label='pasat run 2')
    ax.legend()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('PSD')


# Make the big topo figure
fig = plt.figure(figsize=(14, 9))
axes = iter_topography(info, layout, on_pick=on_pick, fig=fig,
                       axis_facecolor='white', fig_facecolor='white',
                       axis_spinecolor='white')
for ax, ch_idx in axes:
    handles = [
        ax.plot(psds['freqs'], psds['eyesclosed'][ch_idx], color='C0'),
        ax.plot(psds['freqs'], psds['eyesopen'][ch_idx], color='C1'),
        ax.plot(psds['freqs'], psds['pasat_run1'][ch_idx], color='C2'),
        ax.plot(psds['freqs'], psds['pasat_run2'][ch_idx], color='C3'),
    ]
fig.legend(handles)

with open_report(fname.report(subject=args.subject)) as report:
    report.add_figs_to_section(fig, 'PSDs', section='PSDs', replace=True)
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)
