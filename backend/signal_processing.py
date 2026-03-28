import numpy as np
import pandas as pd
from scipy.signal import stft

def generate_spectrograms(df: pd.DataFrame, window_length: int = 60):
    """
    Given a pandas DataFrame of signals, applies a sliding window of size `window_length`
    For each window and each channel, computes the short-time fourier transform spectrogram.
    Returns:
       X: array of shape [num_samples, channels, frequency_bins, time_bins]
    """
    data = df.values # shape: (T, channels)
    T, channels = data.shape
    num_samples = T - window_length
    
    # STFT parameters mapping to the theoretical formulation
    # nperseg controls frequency resolution vs time resolution
    nperseg = 15
    noverlap = 14 # high overlap for fine temporal resolution inside the window
    
    # Calculate dimensions using one dummy run
    dummy_stft = stft(data[:window_length, 0], nperseg=nperseg, noverlap=noverlap)
    freq_bins = dummy_stft[0].shape[0]
    time_bins = dummy_stft[2].shape[1]
    
    X = np.zeros((num_samples, channels, freq_bins, time_bins), dtype=np.float32)
    
    for i in range(num_samples):
        # Extract sliding window
        window_data = data[i : i + window_length, :]
        
        for c in range(channels):
            # Compute STFT
            f, t, Zxx = stft(window_data[:, c], nperseg=nperseg, noverlap=noverlap)
            
            # STFT(t, f) magnitude squared -> Spectrogram
            S = np.abs(Zxx)**2
            
            X[i, c, :, :] = S
            
    return X
