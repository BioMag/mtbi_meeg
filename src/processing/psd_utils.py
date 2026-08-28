"""Power spectral density helpers used by the preprocessing pipeline."""


def compute_welch_psd(raw, *, fmax, n_fft, picks="eeg"):
    """Compute Welch PSD with MNE's Spectrum API.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording.
    fmax : float
        Highest frequency to include.
    n_fft : int
        FFT length. For short recordings this is capped at the number of
        available samples, matching the valid range required by MNE.
    picks : str | list
        Channels passed to ``Raw.compute_psd``.

    Returns
    -------
    psds : ndarray, shape (n_channels, n_frequencies)
        Power spectral densities.
    freqs : ndarray, shape (n_frequencies,)
        Frequency values in Hz.
    """
    if n_fft < 1:
        raise ValueError("n_fft must be a positive integer")
    effective_n_fft = min(int(n_fft), raw.n_times)
    spectrum = raw.compute_psd(
        method="welch", fmax=fmax, n_fft=effective_n_fft, picks=picks, verbose=False
    )
    return spectrum.get_data(), spectrum.freqs
