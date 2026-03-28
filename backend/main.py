from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

# Internal module imports
from data_loader import fetch_financial_data, get_target_prices
from signal_processing import generate_spectrograms
import model as dl_model
import plots

app = FastAPI(title="Financial Time Series CNN API")

# Setup CORS to allow Vite frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the current state (Dataset, Model, etc.)
# Not suitable for production with multiple users, but fine for a demonstration assignment
APP_STATE = {
    'ticker': None,
    'signal_df': None,
    'signal_norm': None,
    'X_tensor': None,
    'y_tensor': None,
    'model': None,
    'history': None
}

class DataRequest(BaseModel):
    ticker: str

@app.post("/api/load_data")
def load_data(req: DataRequest):
    try:
        # 1. Fetch data
        signal_df, signal_norm, _, _ = fetch_financial_data(req.ticker, years=4)
        
        # 2. Extract targets (Next day's close price)
        target = get_target_prices(signal_norm, horizon=1)
        
        # 3. Generate Spectrogram sequences
        # Window size = 60 days
        X = generate_spectrograms(signal_norm, window_length=60)
        
        # Align target to the ends of the windows 
        # Since X[i] contains data from t to t+59, target is t+60.
        y = target.values[60:]
        
        # Check lengths
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Drop rows where target is NaN (due to shifting)
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Store to state
        APP_STATE['ticker'] = req.ticker
        APP_STATE['signal_df'] = signal_df
        APP_STATE['signal_norm'] = signal_norm
        
        APP_STATE['X_tensor'] = torch.tensor(X, dtype=torch.float32)
        APP_STATE['y_tensor'] = torch.tensor(y, dtype=torch.float32)
        
        # Select first valid channel for plotting the single diagram (e.g. Close price is channel 0)
        S_example = X[0, 0, :, :]
        
        return {
            "status": "success",
            "message": f"Data loaded successfully for {req.ticker}. Generated {len(X)} spectrogram windows.",
            "plots": {
                "time_series": plots.plot_time_series(signal_norm, req.ticker),
                "frequency_spectrum": plots.plot_frequency_spectrum(S_example, "Close Channel (1st window)"),
                "spectrogram": plots.plot_spectrogram(S_example, "Close Channel (1st window)")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def train_network():
    if APP_STATE['X_tensor'] is None:
        raise HTTPException(status_code=400, detail="Data must be loaded first.")
        
    X_train = APP_STATE['X_tensor']
    y_train = APP_STATE['y_tensor']
    
    # Simple split (e.g., 80% train, 20% test)
    split_idx = int(0.8 * len(X_train))
    x_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
    x_ts, y_ts = X_train[split_idx:], y_train[split_idx:]
    
    # Train CNN
    # Training briefly for 10 epochs to not block the UI forever
    model, epoch_losses = dl_model.train_model(x_tr, y_tr, epochs=15, lr=0.002)
    APP_STATE['model'] = model
    
    # Predict on Test Set
    preds = dl_model.predict_model(model, x_ts)
    
    # Calculate MSE on test
    mse = float(np.mean((preds - y_ts.numpy())**2))
    
    return {
        "status": "success",
        "mse": mse,
        "plots": {
            "loss_plot": plots.plot_loss(epoch_losses),
            "prediction_plot": plots.plot_predictions(y_ts.numpy(), preds, title="Test Set Prediction")
        }
    }
