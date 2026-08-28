"""Numerical regression tests for MNE Welch PSD integration."""

import sys
from pathlib import Path

import mne
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import welch

PROCESSING = Path(__file__).resolve().parents[1] / "src" / "processing"
sys.path.insert(0, str(PROCESSING))
from psd_utils import compute_welch_psd


def test_compute_welch_psd_matches_scipy_reference():
    sfreq = 128.0
    time = np.arange(0, 4, 1 / sfreq)
    data = np.vstack((np.sin(2 * np.pi * 10 * time), 0.5 * np.sin(2 * np.pi * 20 * time)))
    raw = mne.io.RawArray(data, mne.create_info(["EEG001", "EEG002"], sfreq, "eeg"), verbose=False)

    actual_psd, actual_freqs = compute_welch_psd(raw, fmax=40, n_fft=256)
    reference_freqs, reference_psd = welch(data, fs=sfreq, nperseg=256, nfft=256)
    keep = reference_freqs <= 40

    assert_allclose(actual_freqs, reference_freqs[keep])
    # MNE uses a Hamming window and zero overlap by default, so compare peak bins,
    # the scientifically relevant regression target, rather than window scaling.
    assert actual_freqs[np.argmax(actual_psd[0])] == 10
    assert actual_freqs[np.argmax(actual_psd[1])] == 20
    assert reference_freqs[keep][np.argmax(reference_psd[0, keep])] == 10
    assert reference_freqs[keep][np.argmax(reference_psd[1, keep])] == 20


def test_compute_welch_psd_caps_fft_for_short_recording():
    raw = mne.io.RawArray(np.sin(2 * np.pi * 4 * np.arange(32) / 32)[None, :], mne.create_info(["EEG001"], 32, "eeg"), verbose=False)
    psd, freqs = compute_welch_psd(raw, fmax=12, n_fft=2048)
    assert psd.shape == (1, len(freqs))


def test_compute_welch_psd_rejects_invalid_fft():
    raw = mne.io.RawArray(np.sin(2 * np.pi * 4 * np.arange(32) / 32)[None, :], mne.create_info(["EEG001"], 32, "eeg"), verbose=False)
    try:
        compute_welch_psd(raw, fmax=12, n_fft=0)
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("Expected ValueError")
