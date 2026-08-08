import os
import soundfile as sf
import numpy as np
from scipy import signal
from python_speech_features import mfcc
from pydub import AudioSegment
import tempfile


def convert_to_wav(file_path):
    """
    Converts any audio format (m4a, webm, mp3, etc.) into a temporary .wav file
    using ffmpeg (via pydub). If the file is already a .wav, this still works
    fine -- just re-encodes it cleanly.
    Returns the path to the new temporary .wav file.
    """
    audio = AudioSegment.from_file(file_path)  # ffmpeg auto-detects the format
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name


def load_audio(file_path, sample_rate=22050, duration=3):
    """Loads audio using soundfile (no numba dependency) and resamples if needed."""
    # Convert to a standard wav first -- handles m4a, webm, mp3, etc.
    wav_path = convert_to_wav(file_path)

    y, sr = sf.read(wav_path)

    os.remove(wav_path)  # cleanup the temporary file

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
    """
    y, sr = load_audio(file_path, sample_rate, duration)

    mfcc_features = mfcc(y, samplerate=sr, numcep=13, nfft=1024)
    mfccs_mean = mfcc_features.mean(axis=0)

    freqs, times, spectrogram = signal.spectrogram(y, fs=sr)
    centroid_per_frame = np.sum(freqs[:, None] * spectrogram, axis=0) / (np.sum(spectrogram, axis=0) + 1e-10)
    centroid_mean = centroid_per_frame.mean()

    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    zcr_mean = len(zero_crossings) / len(y)

    feature_vector = np.concatenate([
        mfccs_mean,
        [centroid_mean],
        [zcr_mean],
    ])

    return feature_vector

if __name__ == "__main__":
    test_file = "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    features = extract_features(test_file)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector: {features}")