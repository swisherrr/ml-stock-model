# ML Stock Prediction Model

This module implements a machine learning approach to predict stock price movements 5 days ahead using technical indicators.

## Overview

The model uses XGBoost classifier to predict whether a stock will close higher 5 days in the future (binary classification). It calculates multiple technical indicators, normalizes them to 0-100 range, and uses time-series cross-validation to train the model.

## Features

### Technical Indicators Calculated:
- **RSI** (Relative Strength Index) - 0-100 range
- **MFI** (Money Flow Index) - 0-100 range
- **MACD** (Moving Average Convergence Divergence) - normalized to 0-100
- **Bollinger Bands** - %B normalized to 0-100
- **Stochastic Oscillator** (%K and %D) - 0-100 range
- **ADX** (Average Directional Index) - 0-100 range
- **ATR** (Average True Range) - normalized to 0-100
- **Volume Indicators** (Volume Ratio, OBV) - normalized
- **Price Features** (Price changes, SMA ratios, etc.) - normalized

### Model Features:
- XGBoost classifier with class weighting for imbalanced data
- Time-series cross-validation (no random shuffling)
- Probability predictions (predict_proba)
- Comprehensive evaluation metrics (Precision, Recall, F1, ROC-AUC)
- Feature importance analysis

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost yfinance python-dotenv
```

## Usage

### Training the Model

```bash
cd ML
python main.py
```

This will:
1. Check cache for previously processed data (if available)
2. Fetch data for missing tickers and calculate all technical indicators
3. Cache processed data for future runs
4. Create features and labels (5-day ahead prediction)
5. Train XGBoost model with time-series cross-validation
6. Evaluate on held-out test set
7. Save the trained model to `models/` directory

**Note:** On the first run, all data will be fetched and processed. Subsequent runs will use cached data, making them much faster!

### Cache Management

The processed data (with all indicators calculated) is automatically cached in a SQLite database (`data_cache.db`). This means you don't have to wait for data processing on every run.

**Options:**
- `python main.py` - Uses cache if available (default behavior)
- `python main.py --refresh` - Force refresh: ignore cache and fetch fresh data
- `python main.py --clear-cache` - Clear all cached data

**Benefits:**
- First run: Fetches and processes all data (slower)
- Subsequent runs: Loads from cache (much faster!)
- Only missing or outdated data is re-fetched automatically

### Making Predictions

**Predict for All Stocks (Daily Predictions):**

```bash
python predict_all.py
```

This script will:
- Automatically find the most recent trained model
- Load cached data for all stocks (fast!)
- Generate predictions for all stocks using current day's data
- Display results sorted by probability of going UP

Options:
- `python predict_all.py` - Uses most recent model and cache (default)
- `python predict_all.py --model <path>` - Use a specific model file
- `python predict_all.py --no-cache` - Fetch fresh data instead of using cache

**Predict for Specific Stocks:**

```bash
python predict.py <model_path> <symbol1> [symbol2] ...
```

Example:
```bash
python predict.py models/xgboost_model_20250101_120000.pkl AAPL MSFT TSLA
```

### Using in Python Code

```python
from data_fetcher import fetch_and_prepare_data
from label_generation import prepare_ml_data
from model_trainer import train_and_evaluate
import pickle

# Fetch data and calculate indicators
tickers = ['AAPL', 'MSFT', 'TSLA']
all_data = fetch_and_prepare_data(tickers)

# Prepare ML features and labels
X, y, feature_names = prepare_ml_data(all_data, days_ahead=5)

# Train model
model, results = train_and_evaluate(X, y, feature_names)

# Load saved model
with open('models/xgboost_model_XXX.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    
# Make predictions
probabilities = model.predict_proba(X_test)[:, 1]
```

## Model Output

The trained model outputs:
- **Binary prediction**: 1 if stock predicted to close higher in 5 days, 0 otherwise
- **Probability**: Probability of price increase (0-1 range)

## Evaluation Metrics

The model reports:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Feature Importance**: Top indicators that matter most

## File Structure

```
ML/
├── __init__.py
├── README.md
├── main.py                 # Main training script
├── predict.py              # Prediction script (specific stocks)
├── predict_all.py          # Daily predictions for all stocks
├── data_fetcher.py         # Data fetching and indicator calculation
├── data_cache.py           # Data caching module (SQLite)
├── feature_engineering.py  # Feature normalization
├── label_generation.py     # Label creation
├── model_trainer.py        # Model training and evaluation
├── data_cache.db           # SQLite database with cached data (auto-created)
└── models/                 # Saved models (created after training)
```

## Configuration

Key parameters in `main.py`:
- `days_ahead=5`: Prediction horizon
- `test_size=0.2`: Proportion of data for test set
- `n_splits=5`: Number of CV folds
- `n_estimators=200`: Number of boosting rounds
- `max_depth=6`: Maximum tree depth
- `learning_rate=0.05`: Learning rate

## Notes

- The model uses time-series cross-validation to avoid look-ahead bias
- All indicators are normalized to 0-100 range where applicable
- Class weights are automatically calculated to handle imbalanced data
- The model requires historical data from 2018 onwards for indicator calculation

