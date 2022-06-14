"""
Compute the Power Spectral Density (PSD) for each channel.

"""
#import subprocess
#subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)
import argparse

from mne.io import read_raw_fif
from mne.time_frequency import psd_welch
from mne.externals.h5io import write_hdf5
from mne.viz import iter_topography
from mne import open_report, find_layout, pick_info, pick_types
import matplotlib.pyplot as plt

from config_eeg import fname, n_fft, get_all_fnames, task_from_fname

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# Compute the PSD for each task
psds = dict()

# Maximum frequency at which to compute PSD
fmax = 40



# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom', 'PASAT'] 
bad_subjects = ['01P', '02P', '03P', '04P', '05P', '06P', '07P']#these ica need to be done manually
all_fnames = zip(
    get_all_fnames(args.subject, kind='psds', exclude=exclude),
    get_all_fnames(args.subject, kind='clean', exclude=exclude),
)

for psds_fname, clean_fname in all_fnames:
    task = task_from_fname(clean_fname)
        
    raw = read_raw_fif(fname.clean(subject=args.subject, task=task, run=1),
                       preload=True)
    raw.info['bads']=[]
    clean_1 = raw.copy().crop(tmin=30, tmax=90)
    psds[task+'_1'], freqs = psd_welch(clean_1, fmax=fmax, n_fft=n_fft, picks=['eeg'])
    clean_2 = raw.copy().crop(tmin=120, tmax=180)
    psds[task+'_2'], freqs = psd_welch(clean_2, fmax=fmax, n_fft=n_fft, picks=['eeg'])
    clean_3 = raw.copy().crop(tmin=210, tmax=270)
    psds[task+'_3'], freqs = psd_welch(clean_3, fmax=fmax, n_fft=n_fft, picks=['eeg'])
    
    
    # Add some metadata to the file we are writing
    psds['info'] = raw.info
    psds['freqs'] = freqs
    write_hdf5(fname.psds(subject=args.subject), psds, overwrite=True)
    # TODO: save freqs

# Add a PSD plot to the report.
raw.pick_types(meg=False, eeg=True, eog=False, stim=False, ecg=False, exclude=[])
info = pick_info(raw.info, sel=None)
layout = find_layout(info, exclude=[])


def on_pick(ax, ch_idx):
    """Create a larger PSD plot for when one of the tiny PSD plots is
       clicked."""
    ax.plot(psds['freqs'], psds['ec_1'][ch_idx], color='C0',
            label='eyes closed')
    ax.plot(psds['freqs'], psds['eo_1'][ch_idx], color='C1',
            label='eyes open')
    # ax.plot(psds['freqs'], psds['pasat_run1'][ch_idx], color='C2',
    #         label='pasat run 1')
    # ax.plot(psds['freqs'], psds['pasat_run2'][ch_idx], color='C3',
    #         label='pasat run 2')
    ax.legend()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('PSD')


# Make the big topo figure
fig = plt.figure(figsize=(14, 9))
axes = iter_topography(info, layout, on_pick=on_pick, fig=fig,
                       axis_facecolor='white', fig_facecolor='white',
                       axis_spinecolor='white')
for ax, ch_idx in axes:
    #print(ax)
    handles = [
        ax.plot(psds['freqs'], psds['ec_1'][ch_idx], color='C0'),
        ax.plot(psds['freqs'], psds['eo_1'][ch_idx], color='C1'),
        # ax.plot(psds['freqs'], psds['pasat_run1'][ch_idx], color='C2'),
        # ax.plot(psds['freqs'], psds['pasat_run2'][ch_idx], color='C3'),
    ]
fig.legend(handles)
fig.close()
with open_report(fname.report(subject=args.subject)) as report:
    report.add_figs_to_section(fig, 'PSDs', section='PSDs', replace=True)
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)
