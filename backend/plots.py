import io
import base64
import matplotlib
matplotlib.use('Agg') # Disable interactive GUI since this is a server
import matplotlib.pyplot as plt
import numpy as np

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

def plot_time_series(df, ticker_symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], color='#4361ee')
    ax.set_title(f'Real Time Series Plot: {ticker_symbol}')
    ax.set_ylabel('Normalized Price')
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)

def plot_spectrogram(S_matrix: np.ndarray, channel_name="Close"):
    # Convert energy to dB for better visualization
    # S_matrix represents energy, avoiding log(0)
    S_db = 10 * np.log10(S_matrix + 1e-10)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(S_db, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(f'STFT Spectrogram: {channel_name}')
    ax.set_xlabel('Time Frames (Window Sliding)')
    ax.set_ylabel('Frequency')
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return fig_to_base64(fig)

def plot_frequency_spectrum(S_matrix: np.ndarray, channel_name="Close"):
    # Averaged over time frames to get the overall spectrum characteristics
    S_avg = S_matrix.mean(axis=1)
    S_db = 10 * np.log10(S_avg + 1e-10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(S_db, color='#f72585')
    ax.set_title(f'Frequency Spectrum (Avg over Time): {channel_name}')
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Power (dB)')
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)

def plot_predictions(actual, predicted, title="Prediction vs Actual"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label='Actual Normalized Value', color='black', alpha=0.6, linewidth=1.5)
    ax.plot(predicted, label='Predicted Vector', color='#4cc9f0', alpha=0.9, linewidth=1.5)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='orange')
    ax.set_title('Training MSE Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)
