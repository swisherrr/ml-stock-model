#!/usr/bin/env python3
"""
Main script for ML stock prediction model
Orchestrates data fetching, feature engineering, label generation, and model training
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from data_fetcher import fetch_and_prepare_data
from label_generation import prepare_ml_data
from model_trainer import train_and_evaluate
from data_cache import DataCache


def get_ticker_list():
    """
    Get the hardcoded list of tickers
    Paste your ticker list in the TICKER_LIST below
    """
    # PASTE YOUR TICKER LIST HERE (replace the example list below)
    TICKER_LIST = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO', 'TSM', 'TSLA', 'JPM', 'WMT', 'ORCL', 'LLY', 'V', 'MA', 'NFLX', 'XOM', 'COST', 'JNJ', 'PLTR', 'HD', 'BAC', 'PG', 'SAP', 'ABBV', 'KO', 'GE', 'AMD', 'BABA', 'ASML', 'CSCO', 'TMUS', 'CVX', 'WFC', 'PM', 'CRM', 'IBM', 'TM', 'MS', 'AZN', 'UNH', 'NVS', 'CCZ', 'GS', 'ABT', 'INTU', 'LIN', 'SHEL', 'MCD', 'DIS', 'HSBC', 'RTX', 'NVO', 'AXP', 'CAT', 'NOW', 'MRK', 'T', 'HDB', 'PEP', 'UBER', 'RY', 'GEV', 'BKNG', 'SCHW', 'TMO', 'C', 'ISRG', 'BLK', 'SPGI', 'BA', 'MUFG', 'ACN', 'TBB', 'TXN', 'QCOM', 'PDD', 'AMGN', 'SHOP', 'BSX', 'ANET', 'ADBE', 'ETN', 'SYK', 'ARM', 'SONY', 'NEE', 'AMAT', 'UL', 'DE', 'PGR', 'DHR', 'HON', 'GILD', 'TJX', 'TTE', 'PFE', 'APP', 'UNP', 'KKR', 'APH', 'BHP', 'SAN', 'SPOT', 'BX', 'ADP', 'TD', 'LOW', 'CMCSA', 'MU', 'LRCX', 'YMM', 'COP', 'MELI', 'IBN', 'UBS', 'BTI', 'VRTX', 'KLAC', 'PANW', 'MDT', 'SNY', 'MSTR', 'CRWD', 'ADI', 'IBKR', 'BN', 'WELL', 'NKE', 'CEG', 'RACE', 'CB', 'DASH', 'ICE', 'MO', 'SO', 'BUD', 'SBUX', 'CME', 'CDNS', 'BAM', 'PLD', 'ENB', 'SNPS', 'LMT', 'SMFG', 'MMC', 'TT', 'AMT', 'BBVA', 'COIN', 'RELX', 'DUK', 'PH', 'RBLX', 'SE', 'MCO', 'WM', 'HOOD', 'TRI', 'TDG', 'DELL', 'CTAS', 'BMY', 'MCK', 'INTC', 'RCL', 'BP', 'HCA', 'ORLY', 'GD', 'MDLZ', 'CVNA', 'APO', 'ABNB', 'NOC', 'NTES', 'SHW', 'COF', 'PBR', 'EMR']

    
    # Remove duplicates while preserving order
    seen = set()
    tickers = []
    for ticker in TICKER_LIST:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    
    return tickers


def main(force_refresh=False):
    """
    Main function to run the complete ML pipeline
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh data
    """
    print("=" * 80)
    print("ML Stock Prediction Model - 5 Day Ahead Prediction")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Get ticker list
    print("\n1. Getting ticker list...")
    tickers = get_ticker_list()
    print(f"   Found {len(tickers)} tickers")
    
    # Show cache status
    if not force_refresh:
        cache = DataCache()
        cache_info = cache.get_cache_info()
        if cache_info['symbol_count'] > 0:
            print(f"\n   Cache available: {cache_info['symbol_count']} symbols")
            print(f"   Using cached data (use --refresh to force update)")
    
    # Fetch and prepare data
    print("\n2. Fetching stock data and calculating indicators...")
    if force_refresh:
        print("   Force refresh enabled - ignoring cache and fetching fresh data...")
    else:
        print("   Checking cache first, then fetching missing data...")
    print("   This may take a while depending on the number of tickers...")
    
    all_data = fetch_and_prepare_data(
        symbols=tickers,
        start_date="2018-01-01",
        end_date=None,
        delay=0.2,  # Delay between API calls
        use_cache=True,
        force_refresh=force_refresh
    )
    
    if not all_data:
        print("Error: No data was successfully fetched. Exiting.")
        return
    
    print(f"\n   Successfully processed {len(all_data)} stocks")
    
    # Prepare ML data with features and labels
    print("\n3. Preparing ML features and labels...")
    X, y, feature_names = prepare_ml_data(all_data, days_ahead=5)
    
    if len(X) == 0:
        print("Error: No valid samples after feature preparation. Exiting.")
        return
    
    # Train and evaluate model
    print("\n4. Training and evaluating model...")
    model, results = train_and_evaluate(
        X=X,
        y=y,
        feature_names=feature_names,
        test_size=0.2,
        n_splits=5,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        save_model=True,
        model_dir='models'
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"\nCross-Validation Results:")
    cv = results['cv_results']
    print(f"  Precision: {sum(cv['precision'])/len(cv['precision']):.4f}")
    print(f"  Recall: {sum(cv['recall'])/len(cv['recall']):.4f}")
    print(f"  F1-Score: {sum(cv['f1_score'])/len(cv['f1_score']):.4f}")
    print(f"  ROC-AUC: {sum(cv['roc_auc'])/len(cv['roc_auc']):.4f}")
    
    print(f"\nTest Set Results:")
    test = results['test_metrics']
    print(f"  Precision: {test['precision']:.4f}")
    print(f"  Recall: {test['recall']:.4f}")
    print(f"  F1-Score: {test['f1_score']:.4f}")
    print(f"  ROC-AUC: {test['roc_auc']:.4f}")
    
    print("\n" + "=" * 80)
    print("Model training complete!")
    print("=" * 80)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML stock prediction model')
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force refresh: ignore cache and fetch fresh data'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached data before running'
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        cache = DataCache()
        cache.clear_cache()
        print("Cache cleared. Exiting.")
        sys.exit(0)
    
    model, results = main(force_refresh=args.refresh)

