#!/usr/bin/env python3
"""
Daily prediction script for all stocks
Loads the most recent trained model and generates predictions for all stocks
using the current day's data with cached indicators
"""

import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import fetch_and_prepare_data
from feature_engineering import prepare_features, get_feature_columns
from data_cache import DataCache


def get_ticker_list():
    """
    Get the hardcoded list of tickers (same as main.py)
    """
    TICKER_LIST = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO', 'TSM', 'TSLA', 'JPM', 'WMT', 'ORCL', 'LLY', 'V', 'MA', 'NFLX', 'XOM', 'COST', 'JNJ', 'PLTR', 'HD', 'BAC', 'PG', 'SAP', 'ABBV', 'KO', 'GE', 'AMD', 'BABA', 'ASML', 'CSCO', 'TMUS', 'CVX', 'WFC', 'PM', 'CRM', 'IBM', 'TM', 'MS', 'AZN', 'UNH', 'NVS', 'CCZ', 'GS', 'ABT', 'INTU', 'LIN', 'SHEL', 'MCD', 'DIS', 'HSBC', 'RTX', 'NVO', 'AXP', 'CAT', 'NOW', 'MRK', 'T', 'HDB', 'PEP', 'UBER', 'RY', 'GEV', 'BKNG', 'SCHW', 'TMO', 'C', 'ISRG', 'BLK', 'SPGI', 'BA', 'MUFG', 'ACN', 'TBB', 'TXN', 'QCOM', 'PDD', 'AMGN', 'SHOP', 'BSX', 'ANET', 'ADBE', 'ETN', 'SYK', 'ARM', 'SONY', 'NEE', 'AMAT', 'UL', 'DE', 'PGR', 'DHR', 'HON', 'GILD', 'TJX', 'TTE', 'PFE', 'APP', 'UNP', 'KKR', 'APH', 'BHP', 'SAN', 'SPOT', 'BX', 'ADP', 'TD', 'LOW', 'CMCSA', 'MU', 'LRCX', 'YMM', 'COP', 'MELI', 'IBN', 'UBS', 'BTI', 'VRTX', 'KLAC', 'PANW', 'MDT', 'SNY', 'MSTR', 'CRWD', 'ADI', 'IBKR', 'BN', 'WELL', 'NKE', 'CEG', 'RACE', 'CB', 'DASH', 'ICE', 'MO', 'SO', 'BUD', 'SBUX', 'CME', 'CDNS', 'BAM', 'PLD', 'ENB', 'SNPS', 'LMT', 'SMFG', 'MMC', 'TT', 'AMT', 'BBVA', 'COIN', 'RELX', 'DUK', 'PH', 'RBLX', 'SE', 'MCO', 'WM', 'HOOD', 'TRI', 'TDG', 'DELL', 'CTAS', 'BMY', 'MCK', 'INTC', 'RCL', 'BP', 'HCA', 'ORLY', 'GD', 'MDLZ', 'CVNA', 'APO', 'ABNB', 'NOC', 'NTES', 'SHW', 'COF', 'PBR', 'EMR']
    
    # Remove duplicates while preserving order
    seen = set()
    tickers = []
    for ticker in TICKER_LIST:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    
    return tickers


def find_latest_model(model_dir='models'):
    """
    Find the most recently created model file
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        Path to the most recent model file
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train a model first using main.py")
    
    # Find all model files
    pattern = os.path.join(model_dir, 'xgboost_model_*.pkl')
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in '{model_dir}'. Please train a model first using main.py")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    return model_files[0]


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


def predict_stock_current(symbol, model, feature_names, all_data):
    """
    Predict probability for a stock using current day's data
    
    Args:
        symbol: Stock ticker symbol
        model: Trained XGBoost model
        feature_names: List of feature names used in training
        all_data: Dictionary with processed data (from cache or fresh fetch)
    
    Returns:
        Dictionary with prediction results or None if failed
    """
    if symbol not in all_data:
        return None
    
    try:
        # Get data with indicators (already calculated)
        data_with_indicators = all_data[symbol]
        
        # Prepare features
        features_df = prepare_features(data_with_indicators)
        
        # Get feature columns
        feature_cols = get_feature_columns(features_df)
        
        # Select only features used in training
        X = features_df[feature_cols].copy()
        
        # Remove rows with NaN
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        
        if len(X) == 0:
            return None
        
        # Make predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        
        # Get most recent (latest) prediction
        latest_idx = len(X) - 1
        latest_date = features_df.index[valid_mask][latest_idx]
        latest_close = data_with_indicators.loc[latest_date, 'Close']
        latest_prob = probabilities[latest_idx]
        latest_pred = predictions[latest_idx]
        
        return {
            'symbol': symbol,
            'date': latest_date,
            'close_price': float(latest_close),
            'probability_up': float(latest_prob),
            'prediction': int(latest_pred)
        }
    
    except Exception as e:
        print(f"  Error predicting {symbol}: {e}")
        return None


def predict_all_stocks(model_path=None, use_cache=True):
    """
    Predict for all stocks using current day's data
    
    Args:
        model_path: Path to model file (if None, finds latest)
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        DataFrame with predictions for all stocks
    """
    print("=" * 80)
    print("Daily Stock Predictions - All Stocks")
    print("=" * 80)
    
    # Find latest model if not provided
    if model_path is None:
        print("\n1. Finding latest trained model...")
        model_path = find_latest_model()
        print(f"   Using model: {os.path.basename(model_path)}")
    else:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return None
        print(f"\n1. Loading model: {os.path.basename(model_path)}")
    
    # Load model
    print("   Loading model...")
    model_data = load_model(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    print(f"   Model loaded successfully")
    
    # Get ticker list
    print("\n2. Getting ticker list...")
    tickers = get_ticker_list()
    print(f"   Found {len(tickers)} stocks to predict")
    
    # Fetch data (will use cache if available, but fetch latest if needed)
    print("\n3. Fetching stock data (using cache if available)...")
    print("   Note: If cache is outdated, latest data will be fetched automatically")
    print("   This may take a moment...")
    
    all_data = fetch_and_prepare_data(
        symbols=tickers,
        start_date="2018-01-01",  # Need historical data for indicators
        end_date=None,  # Get up to current day (will check cache freshness)
        delay=0.1,  # Shorter delay since we're using cache
        use_cache=use_cache,
        force_refresh=False  # Use cache but allow automatic updates for latest data
    )
    
    if not all_data:
        print("Error: No data available. Exiting.")
        return None
    
    print(f"   Successfully loaded data for {len(all_data)} stocks")
    
    # Make predictions for all stocks
    print("\n4. Generating predictions for all stocks...")
    print("   Processing each stock...")
    
    predictions = []
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(tickers, 1):
        print(f"   [{i}/{len(tickers)}] {symbol}...", end=" ", flush=True)
        
        result = predict_stock_current(symbol, model, feature_names, all_data)
        
        if result:
            predictions.append(result)
            successful += 1
            print("✓")
        else:
            failed += 1
            print("✗")
    
    if not predictions:
        print("\nError: No successful predictions generated.")
        return None
    
    # Create DataFrame with results
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('probability_up', ascending=False)
    
    print(f"\n   Successfully predicted: {successful} stocks")
    if failed > 0:
        print(f"   Failed: {failed} stocks")
    
    return results_df


def display_results(results_df):
    """
    Display prediction results in a formatted table
    
    Args:
        results_df: DataFrame with predictions
    """
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS - Current Day")
    print("=" * 80)
    print("Probability that each stock will go UP in 5 days:")
    print("-" * 80)
    print(f"{'Stock':<10} {'Date':<12} {'Close Price':<15} {'Prob. UP':<12} {'Prediction':<10}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        ticker = str(row['symbol'])
        date = pd.to_datetime(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
        close = f"${row['close_price']:.2f}"
        prob = float(row['probability_up'])
        pred = "UP" if int(row['prediction']) == 1 else "DOWN"
        
        print(f"{ticker:<10} {date:<12} {close:<15} {prob:.2%}      {pred:<10}")
    
    print("-" * 80)
    print(f"\nTotal stocks with predictions: {len(results_df)}")
    
    # Summary statistics
    avg_prob = results_df['probability_up'].mean()
    up_predictions = (results_df['prediction'] == 1).sum()
    down_predictions = (results_df['prediction'] == 0).sum()
    
    print(f"\nSummary:")
    print(f"  Average probability of UP: {avg_prob:.2%}")
    print(f"  Stocks predicted to go UP: {up_predictions}")
    print(f"  Stocks predicted to go DOWN: {down_predictions}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate daily predictions for all stocks')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (default: uses most recent model)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached data (will fetch fresh data - slower)'
    )
    
    args = parser.parse_args()
    
    try:
        # Generate predictions
        results_df = predict_all_stocks(
            model_path=args.model,
            use_cache=not args.no_cache
        )
        
        if results_df is None:
            print("\nError: Failed to generate predictions.")
            sys.exit(1)
        
        # Display results
        display_results(results_df)
        
        print("\n" + "=" * 80)
        print("Predictions complete!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

