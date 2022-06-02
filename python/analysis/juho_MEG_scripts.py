import mne
from scipy import signal
import numpy as np

filename = ... # cleaded .fif datafile
freqs_filename = ... # f.txt file, this includes the 21 frequency bands
save_filename = ... # a path to power spectra to be saved. Include .npy

raw_clean = mne.io.read_raw_fif(filename, preload = True) # Load file to raw_clean

nfft = 2048 # Length of the FFT
ws = 1 #  Length of sliding window (in seconds)

freqs = np.loadtxt(freqs_filename, delimiter = ",") # Load band limits

fs = raw_clean.info["sfreq"] # Get sampling rate
nperseg = fs * ws # Calculate length of each segment based on sliding window and sampling rate

raw_clean.pick_types(meg = True) # Pick only MEG channels
data = raw_cp.get_data() # Get MEG data as array
    
f, Pxx_den = signal.welch(data, fs, window='hann', nfft = nfft, nperseg = nperseg) # Calculate psd
Pxx_den = 20 * np.log10(Pxx_den) # Convert to decibels
    

# Average psd to frequency bands defined in freqs
arr = np.zeros((Pxx_den.shape[0], freqs.shape[0])) # Generate empty array
for i in range(freqs.shape[0]): # Loop over each band
    f_band = (f > freqs[i,0]) & (f <= freqs[i,1]) # Get the band limits
    arr[:,i] = np.mean(Pxx_den[:, f_band],1) # Insert average PSD from that band to new array we just created

np.save(save_filename, arr)
