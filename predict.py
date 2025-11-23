#!/usr/bin/env python3
"""
Prediction script
Uses trained model to predict probability of stock price increase 5 days ahead
"""

import pickle
import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data, calculate_all_indicators
from feature_engineering import prepare_features, get_feature_columns


def load_model(model_path):
    """
    Load trained model from file
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Dictionary with model and metadata
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_stock(symbol, model_path, start_date="2018-01-01", end_date=None):
    """
    Predict probability of stock price increase 5 days ahead
    
    Args:
        symbol: Stock ticker symbol
        model_path: Path to trained model file
        start_date: Start date for data (to calculate indicators)
        end_date: End date for data
    
    Returns:
        Dictionary with predictions and probabilities
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model_data = load_model(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Fetch stock data
    print(f"Fetching data for {symbol}...")
    data = fetch_stock_data(symbol, start_date, end_date)
    
    if data is None:
        print(f"Error: Could not fetch data for {symbol}")
        return None
    
    # Calculate indicators
    print("Calculating indicators...")
    data_with_indicators = calculate_all_indicators(data)
    
    # Prepare features
    print("Preparing features...")
    features_df = prepare_features(data_with_indicators)
    
    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    
    # Select only features used in training
    X = features_df[feature_cols].copy()
    
    # Remove rows with NaN
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    dates = features_df.index[valid_mask]
    
    if len(X) == 0:
        print("Error: No valid samples after feature preparation")
        return None
    
    # Make predictions
    print("Making predictions...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': dates,
        'close_price': data_with_indicators.loc[dates, 'Close'],
        'prediction': predictions,
        'probability_up': probabilities
    })
    
    # Get most recent prediction
    latest = results.iloc[-1]
    
    print(f"\n{'='*60}")
    print(f"Prediction for {symbol}")
    print(f"{'='*60}")
    print(f"Date: {latest['date']}")
    print(f"Close Price: ${latest['close_price']:.2f}")
    print(f"Prediction: {'UP' if latest['prediction'] == 1 else 'DOWN'}")
    print(f"Probability of UP: {latest['probability_up']:.2%}")
    print(f"{'='*60}")
    
    return {
        'symbol': symbol,
        'latest_prediction': latest.to_dict(),
        'all_predictions': results
    }


def predict_multiple_stocks(symbols, model_path, start_date="2018-01-01", end_date=None):
    """
    Predict for multiple stocks
    
    Args:
        symbols: List of stock ticker symbols
        model_path: Path to trained model file
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        Dictionary mapping symbols to prediction results
    """
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        try:
            result = predict_stock(symbol, model_path, start_date, end_date)
            if result:
                results[symbol] = result
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <symbol1> [symbol2] ...")
        print("Example: python predict.py models/xgboost_model_20250101_120000.pkl AAPL MSFT TSLA")
        sys.exit(1)
    
    model_path = sys.argv[1]
    symbols = sys.argv[2:]
    
    if len(symbols) == 1:
        result = predict_stock(symbols[0], model_path)
    else:
        results = predict_multiple_stocks(symbols, model_path)

