import { useState } from 'react';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [loadingData, setLoadingData] = useState(false);
  const [trainingModel, setTrainingModel] = useState(false);
  
  const [appState, setAppState] = useState({
    dataLoaded: false,
    message: '',
    plots: {},
    mse: null,
    trainMessage: '',
    trainPlots: {}
  });

  const handleLoadData = async () => {
    setLoadingData(true);
    setAppState(prev => ({ ...prev, message: 'Fetching financial data from Yahoo Finance...', dataLoaded: false, plots: {}, mse: null, trainPlots: {} }));
    
    try {
      const res = await fetch(`${API_BASE}/load_data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker })
      });
      const data = await res.json();
      
      if (res.ok) {
        setAppState(prev => ({
          ...prev,
          dataLoaded: true,
          message: data.message,
          plots: data.plots
        }));
      } else {
        setAppState(prev => ({ ...prev, message: `Error: ${data.detail}` }));
      }
    } catch (err) {
      setAppState(prev => ({ ...prev, message: `Connection Error: Is the FastAPI backend running?` }));
    }
    setLoadingData(false);
  };

  const handleTrainModel = async () => {
    setTrainingModel(true);
    setAppState(prev => ({ ...prev, trainMessage: 'Training PyTorch CNN... (15 epochs)', mse: null, trainPlots: {} }));
    
    try {
      const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
      const data = await res.json();
      
      if (res.ok) {
        setAppState(prev => ({
          ...prev,
          mse: data.mse,
          trainMessage: 'Training Complete!',
          trainPlots: data.plots
        }));
      } else {
        setAppState(prev => ({ ...prev, trainMessage: `Error: ${data.detail}` }));
      }
    } catch (err) {
      setAppState(prev => ({ ...prev, trainMessage: 'Error connecting to backend during training.' }));
    }
    setTrainingModel(false);
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Financial Time-Series Patterns</h1>
        <p>Combining time-frequency signal processing & Deep Learning for stock prediction.</p>
      </div>

      <div className="glass-card" style={{ textAlign: 'center' }}>
        <h3 style={{ marginBottom: '1rem', color: '#c4b5fd' }}>Step 1: Data Preparation & Signal Processing</h3>
        <div className="input-group">
          <input 
            type="text" 
            value={ticker} 
            onChange={(e) => setTicker(e.target.value)} 
            placeholder="Enter Ticker (e.g. AAPL, MSFT, ^BSESN)"
          />
          <button onClick={handleLoadData} disabled={loadingData || trainingModel}>
            {loadingData ? <div className="spinner"></div> : 'Generate Spectrograms'}
          </button>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>{appState.message}</p>
      </div>

      {appState.dataLoaded && (
        <>
          <div className="glass-card">
            <span className="tag">Time Domain</span>
            <h3>Time Series Data</h3>
            <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '1rem' }}>Original stock data represented as normalized amplitude over time.</p>
            {appState.plots.time_series && (
              <div className="plot-container">
                <img src={appState.plots.time_series} alt="Time Series Plot" />
              </div>
            )}
          </div>

          <div className="grid-2">
            <div className="glass-card">
              <span className="tag">Frequency Domain</span>
              <h3>Frequency Spectrum</h3>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Fourier Transform representation showing overall frequency components.</p>
              {appState.plots.frequency_spectrum && (
                <div className="plot-container">
                  <img src={appState.plots.frequency_spectrum} alt="Frequency Spectrum Plot" />
                </div>
              )}
            </div>
            <div className="glass-card" style={{ borderColor: 'rgba(139, 92, 246, 0.4)', boxShadow: '0 0 20px rgba(139,92,246,0.1)' }}>
              <span className="tag" style={{ background: 'var(--accent)', color: 'white' }}>Time-Frequency Domain</span>
              <h3>STFT Spectrogram</h3>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Energy distribution in time-frequency. Sliding window approach exposes hidden patterns.</p>
              {appState.plots.spectrogram && (
                <div className="plot-container">
                  <img src={appState.plots.spectrogram} alt="Spectrogram Plot" />
                </div>
              )}
            </div>
          </div>

          <div className="glass-card">
            <span className="tag">Deep Learning Pipeline</span>
            <h3>Architectural Diagram: 2D Convolutional Neural Network</h3>
            <div className="cnn-arch">
              <div className="cnn-layer">
                <span className="cnn-title">Input</span>
                <span className="cnn-detail">Ch: 5 | STFT Matrix</span>
              </div>
              <span className="cnn-arrow">➜</span>
              <div className="cnn-layer">
                <span className="cnn-title">Conv Layer 1</span>
                <span className="cnn-detail">16 Filters | 3x3 | ReLU</span>
                <span className="cnn-detail">MaxPool 2x2</span>
              </div>
              <span className="cnn-arrow">➜</span>
              <div className="cnn-layer">
                <span className="cnn-title">Conv Layer 2</span>
                <span className="cnn-detail">32 Filters | 3x3 | ReLU</span>
                <span className="cnn-detail">MaxPool 2x2</span>
              </div>
              <span className="cnn-arrow">➜</span>
              <div className="cnn-layer">
                <span className="cnn-title">Flatten & Dense</span>
                <span className="cnn-detail">FC 64 | ReLU</span>
              </div>
              <span className="cnn-arrow">➜</span>
              <div className="cnn-layer">
                <span className="cnn-title">Output</span>
                <span className="cnn-detail">Price Predict (1)</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem', flexDirection: 'column', alignItems: 'center' }}>
              <button 
                onClick={handleTrainModel} 
                disabled={trainingModel || loadingData}
                style={{ padding: '1rem 3rem', fontSize: '1.2rem', background: 'linear-gradient(to right, #ec4899, #8b5cf6)' }}
              >
                {trainingModel ? <div className="spinner"></div> : 'Train Model & Predict'}
              </button>
              <p style={{ marginTop: '1rem', color: '#94a3b8' }}>{appState.trainMessage}</p>
            </div>
          </div>
          
          {appState.mse !== null && (
            <div className="grid-2">
              <div className="glass-card">
                <span className="tag" style={{ background: '#10b981', color: 'white' }}>Analysis</span>
                <h3>Training Loss</h3>
                <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Epoch vs Mean Squared Error.</p>
                {appState.trainPlots.loss_plot && (
                  <div className="plot-container">
                    <img src={appState.trainPlots.loss_plot} alt="Training Loss" />
                  </div>
                )}
              </div>
              <div className="glass-card">
                <span className="tag" style={{ background: '#3b82f6', color: 'white' }}>Results</span>
                <h3>Prediction vs Actual</h3>
                <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Test Set MSE: <strong>{appState.mse.toFixed(5)}</strong></p>
                {appState.trainPlots.prediction_plot && (
                  <div className="plot-container">
                    <img src={appState.trainPlots.prediction_plot} alt="Prediction vs Actual" />
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;
