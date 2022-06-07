"""
These are all the relevant parameters that are unique to the EEG analysis
pipeline.
"""

# Some relevant files are in the parent folder.
import sys
sys.path.append('../')

from fnames import FileNames

from config_common import (raw_data_dir, processed_data_dir, figures_dir,
                           reports_dir, all_subjects, tasks)


###############################################################################
# Parameters that should be mentioned in the paper

# Highpass filter above 1Hz. This is needed for the ICA to perform well
# later on. Lowpass filter below 100Hz to get rid of the signal produced by
# the cHPI coils. Notch filters at 50Hz and 100Hz to get rid of powerline.
fmin = 1
fmax = 40
fnotch = [50, 100]

# Computation of the PSDs
n_fft = 1024  # Higher number means more resolution at the lower frequencies


###############################################################################
# Parameters pertaining to the subjects

# Subjects removed from the MEG analysis because of some problem
# TODO: check these 
bad_subjects = [
    24, 29,  # No EEG data present
]

# Analysis is performed on these subjects
subjects = [subject for subject in all_subjects if subject not in bad_subjects]

# Bad MEG channels for each subject.
# Manually marked by Marijn van Vliet.
# TODO: these need to be updated based on Hanna's excel!
bads = {
    1: ['EEG016', 'EEG017', 'EEG004'],
    2: ['EEG003', 'EEG013', 'EEG022', 'EEG027', 'EEG028', 'EEG033', 'EEG060', 'EEG063'],
    3: ['EEG001', 'EEG005', 'EEG025', 'EEG030', 'EEG032', 'EEG037', 'EEG038', 'EEG027', 'EEG023', 'EEG022'],
    4: ['EEG032', 'EEG019', 'EEG018'],
    5: [],
    6: ['EEG013', 'EEG014', 'EEG032', 'EEG033', 'EEG026', 'EEG023', 'EEG022', 'EEG056'],
    7: ['EEG008'],
    8: ['EEG004', 'EEG007', 'EEG014', 'EEG016', 'EEG017', 'EEG020', 'EEG023', 'EEG024', 'EEG025', 'EEG027', 'EEG032', 'EEG033', 'EEG042', 'EEG063'],
    9: ['EEG009', 'EEG032', 'EEG024', 'EEG035', 'EEG006'],
    10: ['EEG004', 'EEG034', 'EEG036', 'EEG043'],
    11: ['EEG024', 'EEG032', 'EEG036'],
    12: ['EEG034', 'EEG035', 'EEG044', 'EEG062', 'EEG055'],
    13: [],
    14: ['EEG024', 'EEG043', 'EEG048'],
    15: ['EEG046', 'EEG044', 'EEG056', 'EEG059', 'EEG060', 'EEG063'],
    16: ['EEG017', 'EEG036', 'EEG025', 'EEG026'],
    17: ['EEG017', 'EEG016', 'EEG032', 'EEG036', 'EEG042', 'EEG024'],
    18: ['EEG017', 'EEG013', 'EEG009', 'EEG001', 'EEG002', 'EEG005', 'EEG006', 'EEG007', 'EEG003'],
    19: ['EEG016', 'EEG017', 'EEG020', 'EEG022', 'EEG023', 'EEG026', 'EEG032', 'EEG033', 'EEG048', 'EEG015', 'EEG021'],
    20: [],
    21: [],
    22: ['EEG014', 'EEG021', 'EEG035', 'EEG036', 'EEG020'],
    23: ['EEG017', 'EEG001', 'EEG028', 'EEG027', 'EEG020', 'EEG019', 'EEG029'],
    24: ['EEG016'],
    25: ['EEG025', 'EEG033', 'EEG026', 'EEG023'],
    26: ['EEG008', 'EEG036'],
    27: ['EEG024', 'EEG032'],
    28: ['EEG003'],
    29: ['EEG040'],
    30: [],
    31: ['EEG029', 'EEG032'],
    32: ['EEG016', 'EEG023', 'EEG033'],
    33: ['EEG007', 'EEG003', 'EEG036', 'EEG032'],
    34: ['EEG034', 'EEG035', 'EEG037', 'EEG050', 'EEG043', 'EEG042', 'EEG048', 'EEG054', 'EEG032', 'EEG033'],
    35: ['EEG032'],
    36: [],
    37: ['EEG020', 'EEG033', 'EEG023', 'EEG055', 'EEG014'],
    38: ['EEG016', 'EEG008', 'EEG023', 'EEG033', 'EEG032', 'EEG031', 'EEG006', 'EEG005'],
    39: ['EEG016', 'EEG023', 'EEG033', 'EEG063'],
    40: ['EEG032', 'EEG044'],
    41: ['EEG006', 'EEG032', 'EEG038', 'EEG028', 'EEG026', 'EEG024', 'EEG027', 'EEG043', 'EEG045', 'EEG056', 'EEG049', 'EEG048', 'EEG047', 'EEG063', 'EEG060', 'EEG062', 'EEG055'],
    42: ['EEG001', 'EEG024', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 'EEG028', 'EEG020'],
    43: ['EEG010'],
    44: ['EEG016', 'EEG036'],
}

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('raw_data_dir', raw_data_dir)
fname.add('processed_data_dir', processed_data_dir)

# Continuous data
fname.add('raw', '{raw_data_dir}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-{task}_run-0{run}_proc-raw_meg.fif')
fname.add('filt', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_filt.fif')
fname.add('clean', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_clean.fif')

# Files used during EOG and ECG artifact suppression
fname.add('ica', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_ica.h5')

# PSD files
fname.add('psds', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_psds.h5')

# Filenames for MNE reports
fname.add('reports_dir', f'{reports_dir}')
fname.add('report', '{reports_dir}/sub-{subject}-report.h5')
fname.add('report_html', '{reports_dir}/sub-{subject}-report.html')

# Filenames for figures
fname.add('figures_dir', f'{figures_dir}')
fname.add('figure_psds', '{figures_dir}/psds.pdf')


def get_all_fnames(subject, kind, exclude=None):
    """Get all filenames for a given subject of a given kind.

    Not all subjects have exactly the same files. For example, subject 1 does
    not have an emptyroom recording, while subject 4 has 2 runs of emptyroom.
    Use this function to get a list of the the files that are present for a
    given subject. It will check which raw files there are and based on that
    will generate a list with corresponding filenames of the given kind.

    You can exclude the recordings for one or more tasks with the ``exclude``
    parameter. For example, to skip the emptyroom recordings, set
    ``exclude='emptyroom'``.

    Parameters
    ----------
    subject : str
        The subject to get the names of the raw files for.
    kind : 'raw' | 'tsss' | 'filt' | 'eog_ecg_events' | 'ica'
        The kind of files to return the filenames for.
    exclude : None | str | list of str
        The tasks to exclude from the list.
        Defaults to not excluding anything.

    Returns
    -------
    all_fnames : list of str
        The names of the files of the given kind.
    """
    import os.path as op

    if exclude is None:
        exclude = []
    elif type(exclude) == str:
        exclude = [exclude]
    elif type(exclude) != list:
        raise TypeError('The `exclude` parameter should be None, str or list')

    all_fnames = list()
    #print('Looking for: ' + str(fname.raw(subject=subject)))
    for task in tasks:
        if task in exclude:
            continue
        for run in [1, 2]:
            if op.exists(fname.raw(subject=subject, task=task, run=run)):
                all_fnames.append(fname.files()[f'{kind}'](subject=subject, task=task, run=run))
    return all_fnames


def task_from_fname(fname):
    """Extract task name from a BIDS filename."""
    import re
    match = re.search(r'task-([^_]+)_run-(\d\d)', str(fname))
    task = match.group(1)
    run = int(match.group(2))
    if task == 'PASAT':
        return f'{task}_run{run}'
    else:
        return task
