"""
Compute the Power Spectral Density (PSD) for each channel.

Running:
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)

TODO: FOOOFing? https://fooof-tools.github.io/fooof/auto_tutorials/index.html
"""

import argparse

from mne.io import read_raw_fif
from mne.time_frequency import psd_array_welch
from h5io import write_hdf5
from mne.viz import iter_topography
from mne import open_report, find_layout, pick_info, pick_types, set_log_level
import matplotlib.pyplot as plt
import datetime
import time

# Save time of beginning of the execution to measure running time
start_time = time.time()


from config_eeg import fname, n_fft, get_all_fnames, task_from_fname

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

# Compute the PSD for each task
psds = dict()

# Maximum frequency at which to compute PSD
fmax = 90



# Not all subjects have files for all conditions. These functions grab the
# files that do exist for the subject.
exclude = ['emptyroom'] 
bad_subjects = ['01P', '02P', '03P', '04P', '05P', '06P', '07P']#these ica need to be done manually
all_fnames = zip(
    get_all_fnames(args.subject, kind='psds', exclude=exclude),
    get_all_fnames(args.subject, kind='clean', exclude=exclude),
)

for psds_fname, clean_fname in all_fnames:
    task = task_from_fname(clean_fname)
    run = 1
    if '1' in task:
        task_wo_run = task.removesuffix('_run1')
    elif '2' in task:
        task_wo_run = task.removesuffix('_run2')    
        run = 2
    else:
        task_wo_run = task
    
    raw = read_raw_fif(fname.clean(subject=args.subject, task=task_wo_run, run=run,ses='01'),
                       preload=True)
    
    # Reduce logging level (technically, one could define it in the read_raw_fif function, but it seems to be buggy)
    # More info about the bug can be found here: https://github.com/mne-tools/mne-python/issues/8872
    set_log_level(verbose='Warning')
    
    raw.info['bads']=[]
    sfreq=raw.info['sfreq']
    
    if 'eo' in task or 'ec' in task:
        clean_1 = raw.copy().crop(tmin=30, tmax=90)
        clean_2 = raw.copy().crop(tmin=120, tmax=180)
        clean_3 = raw.copy().crop(tmin=210, tmax=260)
        #TODO: why this?
        psds[task+'_3'], freqs = psd_array_welch(clean_3.get_data(picks=['eeg']), sfreq=sfreq, 
                                                 fmax=fmax, n_fft=n_fft)

    elif 'PASAT' in task:
        clean_1 = raw.copy().crop(tmin=2, tmax=62)
        clean_2 = raw.copy().crop(tmin=62, tmax=122)
    
    psds[task+'_1'], freqs = psd_array_welch(clean_1.get_data(picks=['eeg']), sfreq=sfreq,
                                             fmax=fmax, n_fft=n_fft)
    psds[task+'_2'], freqs = psd_array_welch(clean_2.get_data(picks=['eeg']), sfreq=sfreq,
                                             fmax=fmax, n_fft=n_fft)
    
    
    # Add some metadata to the file we are writing
    psds['info'] = raw.info
    psds['freqs'] = freqs
    write_hdf5(fname.psds(subject=args.subject, ses='01'), psds, overwrite=True)
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
    ax.plot(psds['freqs'], psds['PASAT_run1_1'][ch_idx], color='C2',
            label='pasat run 1')
    ax.plot(psds['freqs'], psds['PASAT_run2_1'][ch_idx], color='C3',
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
    #print(ax)
    handles = [
        ax.plot(psds['freqs'], psds['ec_1'][ch_idx], color='C0'),
        ax.plot(psds['freqs'], psds['eo_1'][ch_idx], color='C1'),
        ax.plot(psds['freqs'], psds['PASAT_run1_1'][ch_idx], color='C2'),
        ax.plot(psds['freqs'], psds['PASAT_run2_1'][ch_idx], color='C3'),
    ]
fig.legend(handles)


with open_report(fname.report(subject=args.subject)) as report:
    report.add_figure(fig, 'PSDs', replace=True)
    report.save(fname.report_html(subject=args.subject),
                overwrite=True, open_browser=False)

# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time of 03_psds.py is: {round(execution_time,2)} seconds\n')
print('###################################################\n')