import soundfile as sf
import numpy as np
from scipy import signal
from python_speech_features import mfcc


def load_audio(file_path, sample_rate=22050, duration=3):
    """Loads audio using soundfile (no numba dependency) and resamples if needed."""
    y, sr = sf.read(file_path)

    if y.ndim > 1:
        y = y.mean(axis=1)  # stereo -> mono

    if sr != sample_rate:
        num_samples = int(len(y) * sample_rate / sr)
        y = signal.resample(y, num_samples)
        sr = sample_rate

    target_length = sample_rate * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y.astype(np.float32), sr


def extract_features(file_path, sample_rate=22050, duration=3):
    """
    Converts an audio file into a fixed-size numeric feature vector.
    MFCCs via python_speech_features (pure numpy, no numba).
    Spectral centroid + zero-crossing rate computed manually with numpy/scipy.
    """
    y, sr = load_audio(file_path, sample_rate, duration)

    # --- MFCCs: 13 coefficients per audio frame, then averaged over time ---
    mfcc_features = mfcc(y, samplerate=sr, numcep=13, nfft=1024)
    mfccs_mean = mfcc_features.mean(axis=0)  # shape: (13,)

    # --- Spectral centroid: "center of mass" of the frequency spectrum ---
    # Higher = brighter/sharper sound. Computed via FFT manually.
    freqs, times, spectrogram = signal.spectrogram(y, fs=sr)
    centroid_per_frame = np.sum(freqs[:, None] * spectrogram, axis=0) / (np.sum(spectrogram, axis=0) + 1e-10)
    centroid_mean = centroid_per_frame.mean()

    # --- Zero-crossing rate: how often the signal flips sign (noisiness measure) ---
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    zcr_mean = len(zero_crossings) / len(y)

    feature_vector = np.concatenate([
        mfccs_mean,           # 13 numbers
        [centroid_mean],      # 1 number
        [zcr_mean],            # 1 number
    ])

    return feature_vector


if __name__ == "__main__":
    test_file = "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    features = extract_features(test_file)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector: {features}")