#!/usr/bin/env python3
"""
Aggregate Analysis Script
Runs XGBoost, Random Forest, and LightGBM models on the same data
and compares their performance
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from data_fetcher import fetch_and_prepare_data
from label_generation import prepare_ml_data
from data_cache import DataCache

from model_trainer import train_and_evaluate as train_xgboost
from model_random_forest import train_and_evaluate_rf as train_random_forest
from model_lightgbm import train_and_evaluate_lgb as train_lightgbm


def get_ticker_list():
    """
    Get the hardcoded list of tickers
    
    Returns:
        List of ticker symbols
    """
    TICKER_LIST = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO', 'TSM', 'TSLA', 'JPM', 'WMT', 'ORCL', 'LLY', 'V', 'MA', 'NFLX', 'XOM', 'COST', 'JNJ', 'PLTR', 'HD', 'BAC', 'PG', 'SAP', 'ABBV', 'KO', 'GE', 'AMD', 'BABA', 'ASML', 'CSCO', 'TMUS', 'CVX', 'WFC', 'PM', 'CRM', 'IBM', 'TM', 'MS', 'AZN', 'UNH', 'NVS', 'CCZ', 'GS', 'ABT', 'INTU', 'LIN', 'SHEL', 'MCD', 'DIS', 'HSBC', 'RTX', 'NVO', 'AXP', 'CAT', 'NOW', 'MRK', 'T', 'HDB', 'PEP', 'UBER', 'RY', 'GEV', 'BKNG', 'SCHW', 'TMO', 'C', 'ISRG', 'BLK', 'SPGI', 'BA', 'MUFG', 'ACN', 'TBB', 'TXN', 'QCOM', 'PDD', 'AMGN', 'SHOP', 'BSX', 'ANET', 'ADBE', 'ETN', 'SYK', 'ARM', 'SONY', 'NEE', 'AMAT', 'UL', 'DE', 'PGR', 'DHR', 'HON', 'GILD', 'TJX', 'TTE', 'PFE', 'APP', 'UNP', 'KKR', 'APH', 'BHP', 'SAN', 'SPOT', 'BX', 'ADP', 'TD', 'LOW', 'CMCSA', 'MU', 'LRCX', 'YMM', 'COP', 'MELI', 'IBN', 'UBS', 'BTI', 'VRTX', 'KLAC', 'PANW', 'MDT', 'SNY', 'MSTR', 'CRWD', 'ADI', 'IBKR', 'BN', 'WELL', 'NKE', 'CEG', 'RACE', 'CB', 'DASH', 'ICE', 'MO', 'SO', 'BUD', 'SBUX', 'CME', 'CDNS', 'BAM', 'PLD', 'ENB', 'SNPS', 'LMT', 'SMFG', 'MMC', 'TT', 'AMT', 'BBVA', 'COIN', 'RELX', 'DUK', 'PH', 'RBLX', 'SE', 'MCO', 'WM', 'HOOD', 'TRI', 'TDG', 'DELL', 'CTAS', 'BMY', 'MCK', 'INTC', 'RCL', 'BP', 'HCA', 'ORLY', 'GD', 'MDLZ', 'CVNA', 'APO', 'ABNB', 'NOC', 'NTES', 'SHW', 'COF', 'PBR', 'EMR']
    
    seen = set()
    tickers = []
    for ticker in TICKER_LIST:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    
    return tickers


def print_comparison_table(all_results):
    """
    Print a formatted comparison table of all models
    
    Args:
        all_results: Dictionary mapping model names to their results
    """
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - AGGREGATE ANALYSIS")
    print("=" * 100)
    
    print(f"\n{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 70)
    
    print("\nCROSS-VALIDATION RESULTS (Mean ± Std):")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        cv = results['cv_results']
        precision = f"{np.mean(cv['precision']):.4f} ± {np.std(cv['precision']):.4f}"
        recall = f"{np.mean(cv['recall']):.4f} ± {np.std(cv['recall']):.4f}"
        f1 = f"{np.mean(cv['f1_score']):.4f} ± {np.std(cv['f1_score']):.4f}"
        roc_auc = f"{np.mean(cv['roc_auc']):.4f} ± {np.std(cv['roc_auc']):.4f}"
        
        print(f"{model_name:<20} {precision:<18} {recall:<18} {f1:<18} {roc_auc:<18}")
    
    print("\n" + "-" * 70)
    print("TEST SET RESULTS:")
    print("-" * 70)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        test = results['test_metrics']
        print(f"{model_name:<20} {test['precision']:<12.4f} {test['recall']:<12.4f} {test['f1_score']:<12.4f} {test['roc_auc']:<12.4f}")
    
    print("\n" + "-" * 70)
    print("BEST MODEL BY METRIC:")
    print("-" * 70)
    
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics:
        best_model = max(all_results.keys(), 
                        key=lambda x: all_results[x]['test_metrics'][metric])
        best_value = all_results[best_model]['test_metrics'][metric]
        print(f"  {metric.capitalize():<12}: {best_model} ({best_value:.4f})")
    
    print("\n" + "=" * 70)
    best_overall = max(all_results.keys(),
                       key=lambda x: all_results[x]['test_metrics']['f1_score'])
    print(f"BEST OVERALL MODEL (by F1-Score): {best_overall}")
    print("=" * 70)


def print_feature_importance_comparison(all_results, top_n=10):
    """
    Compare top features across all models
    
    Args:
        all_results: Dictionary mapping model names to their results
        top_n: Number of top features to display
    """
    print("\n" + "=" * 100)
    print(f"TOP {top_n} FEATURE IMPORTANCE COMPARISON")
    print("=" * 100)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        fi = results['feature_importance'].head(top_n)
        for i, row in fi.iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")


def save_results_to_csv(all_results, output_dir='results'):
    """
    Save comparison results to CSV files
    
    Args:
        all_results: Dictionary mapping model names to their results
        output_dir: Directory to save CSV files
    
    Returns:
        Tuple of (test_metrics_path, cv_results_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_metrics_data = []
    for model_name, results in all_results.items():
        row = {'model': model_name}
        row.update(results['test_metrics'])
        row.pop('confusion_matrix', None)
        row.pop('predictions', None)
        row.pop('probabilities', None)
        test_metrics_data.append(row)
    
    test_df = pd.DataFrame(test_metrics_data)
    test_path = os.path.join(output_dir, f'model_comparison_{timestamp}.csv')
    test_df.to_csv(test_path, index=False)
    print(f"\nTest metrics saved to: {test_path}")
    
    cv_data = []
    for model_name, results in all_results.items():
        cv = results['cv_results']
        row = {
            'model': model_name,
            'precision_mean': np.mean(cv['precision']),
            'precision_std': np.std(cv['precision']),
            'recall_mean': np.mean(cv['recall']),
            'recall_std': np.std(cv['recall']),
            'f1_score_mean': np.mean(cv['f1_score']),
            'f1_score_std': np.std(cv['f1_score']),
            'roc_auc_mean': np.mean(cv['roc_auc']),
            'roc_auc_std': np.std(cv['roc_auc'])
        }
        cv_data.append(row)
    
    cv_df = pd.DataFrame(cv_data)
    cv_path = os.path.join(output_dir, f'cv_comparison_{timestamp}.csv')
    cv_df.to_csv(cv_path, index=False)
    print(f"CV results saved to: {cv_path}")
    
    return test_path, cv_path


def run_aggregate_analysis(force_refresh=False, use_subset=False):
    """
    Run all three models and compare their performance
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh data
        use_subset: If True, use only first 20 tickers (faster for testing)
    
    Returns:
        Dictionary with results from all models
    """
    print("=" * 100)
    print("AGGREGATE ANALYSIS - Comparing XGBoost, Random Forest, and LightGBM")
    print("=" * 100)
    
    load_dotenv()
    
    print("\n1. Getting ticker list...")
    tickers = get_ticker_list()
    
    if use_subset:
        tickers = tickers[:20]
        print(f"   Using subset of {len(tickers)} tickers for faster testing")
    else:
        print(f"   Found {len(tickers)} tickers")
    
    if not force_refresh:
        cache = DataCache()
        cache_info = cache.get_cache_info()
        if cache_info['symbol_count'] > 0:
            print(f"\n   Cache available: {cache_info['symbol_count']} symbols")
    
    print("\n2. Fetching stock data and calculating indicators...")
    all_data = fetch_and_prepare_data(
        symbols=tickers,
        start_date="2018-01-01",
        end_date=None,
        delay=0.2,
        use_cache=True,
        force_refresh=force_refresh
    )
    
    if not all_data:
        print("Error: No data was successfully fetched. Exiting.")
        return None
    
    print(f"\n   Successfully processed {len(all_data)} stocks")
    
    print("\n3. Preparing ML features and labels...")
    X, y, feature_names, metadata = prepare_ml_data(all_data, days_ahead=5)
    
    if len(X) == 0:
        print("Error: No valid samples after feature preparation. Exiting.")
        return None
    
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(feature_names)}")
    
    common_params = {
        'test_size': 0.2,
        'n_splits': 5,
        'random_state': 42,
        'save_model': True,
        'model_dir': 'models'
    }
    
    all_results = {}
    
    # Train XGBoost
    print("\n" + "=" * 100)
    print("4A. TRAINING XGBOOST MODEL")
    print("=" * 100)
    
    xgb_model, xgb_results = train_xgboost(
        X=X, y=y, feature_names=feature_names,
        n_estimators=200, max_depth=6, learning_rate=0.05,
        **common_params
    )
    all_results['XGBoost'] = xgb_results
    
    # Train Random Forest
    print("\n" + "=" * 100)
    print("4B. TRAINING RANDOM FOREST MODEL")
    print("=" * 100)
    
    rf_model, rf_results = train_random_forest(
        X=X, y=y, feature_names=feature_names,
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        **common_params
    )
    all_results['Random Forest'] = rf_results
    
    # Train LightGBM
    print("\n" + "=" * 100)
    print("4C. TRAINING LIGHTGBM MODEL")
    print("=" * 100)
    
    lgb_model, lgb_results = train_lightgbm(
        X=X, y=y, feature_names=feature_names,
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31,
        **common_params
    )
    all_results['LightGBM'] = lgb_results
    
    # Print comparison
    print_comparison_table(all_results)
    print_feature_importance_comparison(all_results, top_n=10)
    
    # Save results
    save_results_to_csv(all_results)
    
    print("\n" + "=" * 100)
    print("AGGREGATE ANALYSIS COMPLETE!")
    print("=" * 100)
    
    return all_results


def main():
    """Main function to run aggregate analysis"""
    parser = argparse.ArgumentParser(description='Run aggregate analysis on all ML models')
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force refresh: ignore cache and fetch fresh data'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Use only first 20 tickers for faster testing'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached data before running'
    )
    
    args = parser.parse_args()
    
    if args.clear_cache:
        cache = DataCache()
        cache.clear_cache()
        print("Cache cleared.")
        if not args.refresh:
            print("Exiting. Run again without --clear-cache to continue.")
            sys.exit(0)
    
    results = run_aggregate_analysis(
        force_refresh=args.refresh,
        use_subset=args.subset
    )
    
    return results


if __name__ == "__main__":
    main()
