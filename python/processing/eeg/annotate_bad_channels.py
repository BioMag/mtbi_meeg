"""
This script is used for manually annotation bad EEG channels.
The idea is to run this in an ipython console through:

>>> %run annotate_bad_channels.py <subject number>

The script will load the raw data and make a plot of it. Inside the plot you
can scroll through the data and click on the channel names to mark them as bad.
When you close the plot, the script will print out the channels you have
marked. These channels can then be added to the big `bads` list inside
config_eeg.py.
"""
import argparse

import mne
import json
from config_eeg import fname, bads
from config_common import tasks

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()

raw = mne.io.read_raw_fif(fname.tsss(subject=args.subject, ses='01', task=tasks[1], run=1), preload=True)
#raw.info['bads'] = bads[args.subject]
raw.pick_types(meg=True, eeg=False, eog=False, ecg=False)
raw.filter(1, 100)
raw.notch_filter([50, 100])
raw.plot(scalings=dict(eog=100E-6, eeg=50E-6))
print(raw.info['bads'])


chs = raw.info['ch_names']
frontal_picks = mne.read_vectorview_selection('frontal', info=raw.info)
occipital_picks = mne.read_vectorview_selection('occipital', info=raw.info)
temporal_picks = mne.read_vectorview_selection('temporal', info=raw.info)
parietal_picks = mne.read_vectorview_selection('parietal', info=raw.info)

frontal_idxs= [chs.index(f_ch) for f_ch in frontal_picks]
occipital_idxs= [chs.index(o_ch) for o_ch in occipital_picks]
temporal_idxs= [chs.index(t_ch) for t_ch in temporal_picks]
parietal_idxs= [chs.index(p_ch) for p_ch in parietal_picks]
#picks = raw.pick_channels(mne.read_vectorview_selection('frontal', info=raw.info))

ROI_chs = {'frontal':frontal_idxs, 'occipital':occipital_idxs, 'temporal':temporal_idxs, 'parietal':parietal_idxs}

with open('ROI_chs.txt', 'w') as file:
    file.write(json.dumps(ROI_chs))