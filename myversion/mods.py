#!/usr/bin/env python3
import copy
import tempfile

import librosa
import librosa as rs
import numpy as np
import scipy
import soundfile as sf
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from scipy import signal
from scipy.signal import lfilter, resample


def vtln(x, coef=0.0):
    # STFT
    stft_result = rs.core.stft(x)
    mag, phase = rs.magphase(stft_result)
    # Adding a small constant to the magnitude before taking the logarithm
    log_spec = np.log(
        mag + 1e-10
    ).T  # 1e-10 is an arbitrary small number to prevent log(0)
    phase = phase.T

    # Frequency
    freq = np.linspace(0, np.pi, log_spec.shape[1])
    freq_warped = freq + 2.0 * np.arctan(
        coef * np.sin(freq) / (1 - coef * np.cos(freq))
    )

    # Warping
    mag_warped = np.zeros(log_spec.shape, dtype=log_spec.dtype)
    for t in range(log_spec.shape[0]):
        mag_warped[t, :] = np.interp(freq, freq_warped, log_spec[t, :])

    # ISTFT
    y = np.real(rs.core.istft(np.exp(mag_warped).T * phase.T)).astype(x.dtype)

    return y


def modspec_smoothing(x, coef=0.1):
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # STFT
    mag_x, phase_x = rs.magphase(rs.core.stft(x))
    # Prevent log(0) by adding a small constant
    log_mag_x = np.log(mag_x + 1e-10).T
    phase_x = phase_x.T

    # Smoothing trajectory
    mag_x_smoothed = _trajectory_smoothing(log_mag_x, coef)

    # ISTFT
    y = np.real(rs.core.istft(np.exp(mag_x_smoothed).T * phase_x.T)).astype(x.dtype)

    # Normalize safely
    norm_factor = np.sqrt(np.sum(y * y))
    if norm_factor > 0:  # Ensure the denominator is not zero
        y *= np.sqrt(np.sum(x * x)) / norm_factor
    else:
        y *= 0  # If y is silent, just set it to zero

    return y


# resampling
def resampling(x, coef=1.0, fs=16000):
    fn_r, fn_w = tempfile.NamedTemporaryFile(
        mode="r", suffix=".wav"
    ), tempfile.NamedTemporaryFile(mode="w", suffix=".wav")

    sf.write(fn_r.name, x, fs, "PCM_16")
    with WavReader(fn_r.name) as fr:
        with WavWriter(fn_w.name, fr.channels, fr.samplerate) as fw:
            tsm = wsola(
                channels=fr.channels,
                speed=coef,
                frame_length=256,
                synthesis_hop=int(fr.samplerate / 70.0),
            )
            tsm.run(fr, fw)

    y = resample(librosa.load(fn_w.name)[0], len(x)).astype(x.dtype)
    fn_r.close()
    fn_w.close()

    return y


def lpc(signal, order):
    """Compute the Linear Predictive Coefficients.
    Returns the order+1 LPC coefficients for the signal, computed using the
    Levinson-Durbin algorithm.
    """
    # Number of autocorrelation lags
    n = order + 1

    # Compute autocorrelation
    r = np.correlate(signal, signal, mode="full")
    r = r[len(signal) - 1 : len(signal) - 1 + n]

    # Initialize arrays
    a = np.zeros(n)
    E = np.zeros(n)
    k = np.zeros(n)

    # Initialize recursion
    a[0] = 1
    E[0] = r[0]

    # Durbin's recursion
    for i in range(1, n):
        k[i] = -np.dot(a[:i], r[i:0:-1]) / E[i - 1]
        a[i] = k[i]
        for j in range(1, i):
            a[j] = a[j] + k[i] * a[i - j]
        E[i] = (1 - k[i] ** 2) * E[i - 1]

    return a


def vp_baseline2(
    x,
    mcadams=0.8,
    winlen=int(20 * 0.001 * 16000),
    shift=int(10 * 0.001 * 16000),
    lp_order=20,
):
    eps = np.finfo(np.float32).eps
    x2 = copy.deepcopy(x) + eps
    length_x = len(x2)

    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    n_frame = 1 + np.floor((length_x - winlen) / shift).astype(
        int
    )  # nr of complete frames
    y = np.zeros([length_x])

    for m in np.arange(1, n_frame):
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_x))
        frame = x2[index] * win
        a_lpc = lpc(frame, lp_order)
        poles = scipy.signal.tf2zpk([1], a_lpc)[1]
        ind_imag = np.where(np.isreal(poles) == False)[0]
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams

        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi
        new_angles[np.where(new_angles <= 0)] = 0

        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(
                1j * new_angles[k]
            )
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = np.abs(
                poles[ind_imag_con[k] + 1]
            ) * np.exp(-1j * new_angles[k])

        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # get residual excitation for reconstruction
        res = lfilter(a_lpc, np.array(1), frame)
        # reconstruct frames with new lpc coefficient
        frame_rec = lfilter(np.array([1]), a_lpc_new, res)
        frame_rec = frame_rec * win

        outindex = np.arange(m * shift, m * shift + len(frame_rec))
        # overlap add
        y[outindex] = y[outindex] + frame_rec

    y = y / np.max(np.abs(y))
    return y.astype(x.dtype)


def _trajectory_smoothing(x, thresh=0.5):
    y = copy.copy(x)

    b, a = signal.butter(2, thresh)
    for d in range(y.shape[1]):
        y[:, d] = signal.filtfilt(b, a, y[:, d])
        y[:, d] = signal.filtfilt(b, a, y[::-1, d])[::-1]

    return y


def clipping(x, thresh=0.5):
    # Replace non-finite values with zero
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute the histogram on the absolute values of x
    hist, bins = np.histogram(np.abs(x), 1000)
    hist = np.cumsum(hist)

    # Determine the threshold for clipping based on the histogram
    abs_thresh = bins[
        np.where(hist >= min(max(0.0, thresh), 1.0) * np.amax(hist))[0][0]
    ]

    # Clip values that are outside [-abs_thresh, abs_thresh]
    y = np.clip(x, -abs_thresh, abs_thresh)

    # Normalize the clipped waveform to maintain the original energy
    y = y * np.divide(
        np.sqrt(np.sum(x * x)),
        np.sqrt(np.sum(y * y)),
        out=np.zeros_like(np.sqrt(np.sum(x * x))),
        where=np.sqrt(np.sum(y * y)) != 0,
    )

    return y


# chorus effect
def chorus(x, coef=0.1):
    coef = max(0.0, coef)
    xp, xo, xm = vtln(x, coef), vtln(x, 0.0), vtln(x, -coef)

    return (xp + xo + xm) / 3.0
