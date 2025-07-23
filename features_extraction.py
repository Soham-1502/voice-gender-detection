# features_extraction.py
import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    pitch = librosa.yin(y, fmin=50, fmax=300).mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    energy = np.sum(y ** 2) / len(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)

    # New features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()

    return np.hstack([pitch, zcr, energy, centroid, bandwidth, rolloff, mfccs_mean, mfccs_std])
