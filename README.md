# Pattern Recognition for Financial Time Series Forecasting

**Name:** Devika S  
**University Registration No:** TCR24CS024  

## Project Overview

This assignment explores the combination of time–frequency signal processing and deep learning to predict stock prices using financial time series data. 

Financial signals (such as daily stock prices and volume) are non-stationary, meaning their statistical properties fluctuate over time. To capture these changing frequency dynamics, the application transforms the signals into a time-frequency domain.

### Methodology
1. **Data Preparation**: Financial time series data is downloaded using `yfinance` for a given ticker and aligned with an overall market index.
2. **Signal Processing**: The application extracts small sliding windows of the signal and applies a **Short-Time Fourier Transform (STFT)** to generate a 2D **Spectrogram**, mapping signal energy across time and frequency.
3. **Prediction Model & Analysis**: A custom **2D Convolutional Neural Network (CNN)** acts as a regression architecture over these spectrogram representations to learn hidden trends and output future stock price predictions. The results are analyzed using Mean Squared Error (MSE).

---

## Technical Stack
- **Frontend**: React + Vite (Vanilla JS and HTML/CSS for structurally sleek layout and glassmorphism styling features).
- **Backend API**: FastAPI (Python 3.x).
- **Machine Learning Layer**: PyTorch (1D/2D Convolutional models).
- **Signal Processing API**: SciPy (STFT calculations) and Matplotlib (visual mappings).

## Running the Application Locally

The application is split into two modules: a backend machine learning pipeline server and a frontend React Single Page Application (SPA).

### 1. Launch Backend Server
From the root directory, navigate to the backend folder and start the API securely.
```bash
cd backend

# Create Virtual Environment (Optional but recommended)
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Launch Frontend UI
In a separate terminal, navigate to the frontend directory.
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser. From here, you can input a financial ticker (e.g. `AAPL`, `^BSESN`) to generate spectrograms automatically from Yahoo data, and finally start model training on your generated features directly in the browser!