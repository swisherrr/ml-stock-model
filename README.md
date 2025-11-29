# ML Stock Prediction Model

This module implements a machine learning approach to predict stock price movements 5 days ahead using technical indicators.

## Overview

The model uses an ensemble of three classifiers (XGBoost, Random Forest, and LightGBM) to predict whether a stock will close higher 5 days in the future (binary classification). It calculates multiple technical indicators, normalizes them to 0-100 range, and uses time-series cross-validation to train the models. A weighted ensemble combines all three models for potentially improved predictions.

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
- **XGBoost**: Gradient boosting with regularized objective function
- **Random Forest**: Bagging ensemble with bootstrap sampling
- **LightGBM**: Histogram-based gradient boosting with leaf-wise growth
- **Weighted Ensemble**: Combines all three models weighted by CV performance
- Class weighting for imbalanced data (all models)
- Time-series cross-validation (no random shuffling)
- Probability predictions (predict_proba)
- Comprehensive evaluation metrics (Precision, Recall, F1, ROC-AUC)
- Feature importance analysis and consensus across models

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm yfinance python-dotenv
```

## Usage

### Training Individual Models

**XGBoost (Original):**
```bash
cd ML
python main.py
```

**Random Forest:**
```bash
python model_random_forest.py
```

**LightGBM:**
```bash
python model_lightgbm.py
```

Each script will:
1. Check cache for previously processed data (if available)
2. Fetch data for missing tickers and calculate all technical indicators
3. Cache processed data for future runs
4. Create features and labels (5-day ahead prediction)
5. Train the model with time-series cross-validation
6. Evaluate on held-out test set
7. Save the trained model to `models/` directory

### Running Aggregate Analysis (All 3 Models + Ensemble)

```bash
python aggregate_analysis.py
```

This comprehensive script will:
1. Train all three models (XGBoost, Random Forest, LightGBM) on identical data splits
2. Perform time-series cross-validation for each model
3. Calculate ensemble weights based on CV ROC-AUC scores
4. Create a weighted ensemble prediction
5. Compare all models and the ensemble on the test set
6. Analyze feature importance consensus across all models
7. Save results to CSV files in `results/` directory

**Options:**
- `python aggregate_analysis.py` - Run on all stocks (default)
- `python aggregate_analysis.py --subset` - Run on 20-stock subset (faster, for testing)
- `python aggregate_analysis.py --refresh` - Force refresh: ignore cache and fetch fresh data
- `python aggregate_analysis.py --clear-cache` - Clear all cached data

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
from model_random_forest import train_random_forest
from model_lightgbm import train_lightgbm
import pickle
import numpy as np

# Fetch data and calculate indicators
tickers = ['AAPL', 'MSFT', 'TSLA']
all_data = fetch_and_prepare_data(tickers)

# Prepare ML features and labels
X, y, feature_names = prepare_ml_data(all_data, days_ahead=5)

# Train individual models
xgb_model, xgb_results = train_and_evaluate(X, y, feature_names)
rf_model, rf_results = train_random_forest(X, y, feature_names)
lgb_model, lgb_results = train_lightgbm(X, y, feature_names)

# Create weighted ensemble predictions
weights = [0.33, 0.34, 0.33]  # Or calculate from CV scores
ensemble_proba = (
    weights[0] * xgb_model.predict_proba(X_test)[:, 1] +
    weights[1] * rf_model.predict_proba(X_test)[:, 1] +
    weights[2] * lgb_model.predict_proba(X_test)[:, 1]
)

# Load saved model
with open('models/xgboost_model_XXX.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    
# Make predictions
probabilities = model.predict_proba(X_test)[:, 1]
```

## Model Output

Each trained model outputs:
- **Binary prediction**: 1 if stock predicted to close higher in 5 days, 0 otherwise
- **Probability**: Probability of price increase (0-1 range)

The weighted ensemble combines probabilities:
```
P(UP) = (w1 × XGBoost_prob) + (w2 × RF_prob) + (w3 × LightGBM_prob)
```
Where weights are calculated from cross-validation ROC-AUC scores.

## Evaluation Metrics

The models report:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Feature Importance**: Top indicators that matter most
- **Feature Consensus**: Features ranked highly by all three models

## File Structure

```
ML/
├── __init__.py
├── README.md
├── main.py                   # Main XGBoost training script
├── model_random_forest.py    # Random Forest training script
├── model_lightgbm.py         # LightGBM training script
├── aggregate_analysis.py     # Compare all models + ensemble
├── predict.py                # Prediction script (specific stocks)
├── predict_all.py            # Daily predictions for all stocks
├── data_fetcher.py           # Data fetching and indicator calculation
├── data_cache.py             # Data caching module (SQLite)
├── feature_engineering.py    # Feature normalization
├── label_generation.py       # Label creation
├── model_trainer.py          # XGBoost model training and evaluation
├── data_cache.db             # SQLite database with cached data (auto-created)
├── models/                   # Saved models (created after training)
│   ├── xgboost_model_*.pkl
│   ├── random_forest_model_*.pkl
│   └── lightgbm_model_*.pkl
└── results/                  # Analysis results (created by aggregate_analysis.py)
    ├── model_comparison_*.csv
    └── cv_comparison_*.csv
```

## Configuration

### XGBoost Parameters (main.py, model_trainer.py):
- `n_estimators=200`: Number of boosting rounds
- `max_depth=6`: Maximum tree depth
- `learning_rate=0.05`: Learning rate
- `scale_pos_weight`: Auto-calculated for class imbalance

### Random Forest Parameters (model_random_forest.py):
- `n_estimators=200`: Number of trees
- `max_depth=10`: Maximum tree depth
- `min_samples_split=5`: Minimum samples to split a node
- `min_samples_leaf=2`: Minimum samples in a leaf
- `class_weight='balanced'`: Auto-balance classes
- `oob_score=True`: Out-of-bag error estimation
- `n_jobs=-1`: Use all CPU cores

### LightGBM Parameters (model_lightgbm.py):
- `n_estimators=200`: Number of boosting rounds
- `max_depth=6`: Maximum tree depth
- `num_leaves=31`: Maximum number of leaves
- `learning_rate=0.05`: Learning rate
- `scale_pos_weight`: Auto-calculated for class imbalance

### Common Parameters:
- `days_ahead=5`: Prediction horizon
- `test_size=0.2`: Proportion of data for test set
- `n_splits=5`: Number of CV folds

## Notes

- The models use time-series cross-validation to avoid look-ahead bias
- All indicators are normalized to 0-100 range where applicable
- Class weights are automatically calculated to handle imbalanced data
- The models require historical data from 2018 onwards for indicator calculation
- The ensemble weights are calculated from cross-validation performance, giving more weight to better-performing models
- Feature importance consensus across models helps validate which indicators are genuinely predictive
