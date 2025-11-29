#!/usr/bin/env python3
"""
Aggregate Analysis Script
Runs XGBoost, Random Forest, and LightGBM models on the same data,
compares their performance, and creates a weighted ensemble
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import (precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix)

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


def time_series_split(X, y, n_splits=5, test_size=0.2):
    """
    Create time-series cross-validation splits
    No random shuffling - maintains temporal order
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of splits for cross-validation
        test_size: Proportion of data to use for final test set
    
    Returns:
        List of (train_idx, val_idx) tuples for CV, and (train_idx, test_idx) for final split
    """
    n_samples = len(X)
    
    test_start_idx = int(n_samples * (1 - test_size))
    train_indices = np.arange(0, test_start_idx)
    test_indices = np.arange(test_start_idx, n_samples)
    
    cv_splits = []
    train_size = len(train_indices)
    
    for i in range(n_splits):
        fold_train_size = int(train_size * (i + 1) / n_splits)
        fold_train_idx = train_indices[:fold_train_size]
        fold_val_start = fold_train_size
        fold_val_end = min(fold_train_size + int(train_size / n_splits), train_size)
        fold_val_idx = train_indices[fold_val_start:fold_val_end]
        
        if len(fold_val_idx) > 0:
            cv_splits.append((fold_train_idx, fold_val_idx))
    
    return cv_splits, (train_indices, test_indices)


def calculate_ensemble_weights(all_results, weight_metric='roc_auc'):
    """
    Calculate normalized weights for each model based on their CV performance
    
    Args:
        all_results: Dictionary mapping model names to their results
        weight_metric: Metric to use for weighting ('roc_auc', 'f1_score', 'precision', 'recall')
    
    Returns:
        Dictionary mapping model names to their weights
    """
    scores = {}
    for model_name, results in all_results.items():
        cv = results['cv_results']
        scores[model_name] = np.mean(cv[weight_metric])
    
    total_score = sum(scores.values())
    weights = {model: score / total_score for model, score in scores.items()}
    
    return weights


def create_weighted_ensemble_predictions(models, X_test, weights):
    """
    Create weighted ensemble predictions from multiple models
    
    Args:
        models: Dictionary mapping model names to trained models
        X_test: Test features
        weights: Dictionary mapping model names to their weights
    
    Returns:
        Tuple of (ensemble_probabilities, ensemble_predictions)
    """
    ensemble_proba = np.zeros(len(X_test))
    
    for model_name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        ensemble_proba += weights[model_name] * proba
    
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    return ensemble_proba, ensemble_pred


def evaluate_ensemble(y_true, y_pred, y_proba):
    """
    Evaluate ensemble model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
    
    Returns:
        Dictionary with evaluation metrics
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def print_comparison_table(all_results, ensemble_results=None, weights=None):
    """
    Print a formatted comparison table of all models including ensemble
    
    Args:
        all_results: Dictionary mapping model names to their results
        ensemble_results: Dictionary with ensemble evaluation metrics
        weights: Dictionary mapping model names to their weights
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
    
    if ensemble_results:
        print("-" * 70)
        print(f"{'WEIGHTED ENSEMBLE':<20} {ensemble_results['precision']:<12.4f} {ensemble_results['recall']:<12.4f} {ensemble_results['f1_score']:<12.4f} {ensemble_results['roc_auc']:<12.4f}")
    
    if weights:
        print("\n" + "-" * 70)
        print("ENSEMBLE WEIGHTS (based on CV ROC-AUC):")
        print("-" * 70)
        for model_name, weight in weights.items():
            print(f"  {model_name:<20} {weight:.4f} ({weight*100:.2f}%)")
    
    print("\n" + "-" * 70)
    print("BEST MODEL BY METRIC:")
    print("-" * 70)
    
    all_models = dict(all_results)
    if ensemble_results:
        all_models['Weighted Ensemble'] = {'test_metrics': ensemble_results}
    
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics:
        best_model = max(all_models.keys(), 
                        key=lambda x: all_models[x]['test_metrics'][metric])
        best_value = all_models[best_model]['test_metrics'][metric]
        print(f"  {metric.capitalize():<12}: {best_model} ({best_value:.4f})")
    
    print("\n" + "=" * 70)
    best_overall = max(all_models.keys(),
                       key=lambda x: all_models[x]['test_metrics']['f1_score'])
    print(f"BEST OVERALL MODEL (by F1-Score): {best_overall}")
    print("=" * 70)


def print_ensemble_details(ensemble_results, weights):
    """
    Print detailed ensemble results
    
    Args:
        ensemble_results: Dictionary with ensemble evaluation metrics
        weights: Dictionary mapping model names to their weights
    """
    print("\n" + "=" * 100)
    print("WEIGHTED ENSEMBLE DETAILS")
    print("=" * 100)
    
    print("\nModel Weights (based on CV ROC-AUC performance):")
    print("-" * 50)
    for model_name, weight in weights.items():
        bar = "█" * int(weight * 50)
        print(f"  {model_name:<15} {weight:.4f} ({weight*100:.1f}%) {bar}")
    
    print("\nEnsemble Formula:")
    print("-" * 50)
    terms = [f"({w:.3f} × {name})" for name, w in weights.items()]
    print(f"  P(UP) = {' + '.join(terms)}")
    
    print("\nEnsemble Test Set Performance:")
    print("-" * 50)
    print(f"  Precision: {ensemble_results['precision']:.4f}")
    print(f"  Recall:    {ensemble_results['recall']:.4f}")
    print(f"  F1-Score:  {ensemble_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {ensemble_results['roc_auc']:.4f}")
    
    cm = ensemble_results['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
    print(f"    False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")


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


def print_feature_importance_consensus(all_results, top_n=10):
    """
    Find consensus top features across all models
    
    Args:
        all_results: Dictionary mapping model names to their results
        top_n: Number of top features to consider from each model
    """
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE CONSENSUS (Across All Models)")
    print("=" * 100)
    
    feature_scores = {}
    
    for model_name, results in all_results.items():
        fi = results['feature_importance'].head(top_n)
        for rank, (_, row) in enumerate(fi.iterrows()):
            feature = row['feature']
            score = top_n - rank
            if feature not in feature_scores:
                feature_scores[feature] = {'total_score': 0, 'appearances': 0, 'models': []}
            feature_scores[feature]['total_score'] += score
            feature_scores[feature]['appearances'] += 1
            feature_scores[feature]['models'].append(model_name)
    
    sorted_features = sorted(feature_scores.items(), 
                            key=lambda x: (x[1]['appearances'], x[1]['total_score']), 
                            reverse=True)
    
    print(f"\n{'Feature':<30} {'Models':<10} {'Score':<10} {'Appears In'}")
    print("-" * 80)
    
    for feature, data in sorted_features[:top_n]:
        models_str = ', '.join(data['models'])
        print(f"{feature:<30} {data['appearances']:<10} {data['total_score']:<10} {models_str}")


def save_results_to_csv(all_results, ensemble_results, weights, output_dir='results'):
    """
    Save comparison results to CSV files
    
    Args:
        all_results: Dictionary mapping model names to their results
        ensemble_results: Dictionary with ensemble evaluation metrics
        weights: Dictionary mapping model names to their weights
        output_dir: Directory to save CSV files
    
    Returns:
        Tuple of (test_metrics_path, cv_results_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_metrics_data = []
    for model_name, results in all_results.items():
        row = {'model': model_name, 'weight': weights.get(model_name, 0)}
        row.update({k: v for k, v in results['test_metrics'].items() 
                   if k not in ['confusion_matrix', 'predictions', 'probabilities']})
        test_metrics_data.append(row)
    
    ensemble_row = {'model': 'Weighted Ensemble', 'weight': 1.0}
    ensemble_row.update({k: v for k, v in ensemble_results.items() 
                        if k not in ['confusion_matrix', 'predictions', 'probabilities']})
    test_metrics_data.append(ensemble_row)
    
    test_df = pd.DataFrame(test_metrics_data)
    test_path = os.path.join(output_dir, f'model_comparison_{timestamp}.csv')
    test_df.to_csv(test_path, index=False)
    print(f"\nTest metrics saved to: {test_path}")
    
    cv_data = []
    for model_name, results in all_results.items():
        cv = results['cv_results']
        row = {
            'model': model_name,
            'weight': weights.get(model_name, 0),
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
    Run all three models, create weighted ensemble, and compare performance
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh data
        use_subset: If True, use only first 20 tickers (faster for testing)
    
    Returns:
        Dictionary with results from all models and ensemble
    """
    print("=" * 100)
    print("AGGREGATE ANALYSIS - XGBoost, Random Forest, LightGBM + Weighted Ensemble")
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
    
    _, (train_idx, test_idx) = time_series_split(X, y, n_splits=5, test_size=0.2)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = pd.Series(y.iloc[test_idx].values)
    
    common_params = {
        'test_size': 0.2,
        'n_splits': 5,
        'random_state': 42,
        'save_model': True,
        'model_dir': 'models'
    }
    
    all_results = {}
    trained_models = {}
    
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
    trained_models['XGBoost'] = xgb_model
    
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
    trained_models['Random Forest'] = rf_model
    
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
    trained_models['LightGBM'] = lgb_model
    
    # Create Weighted Ensemble
    print("\n" + "=" * 100)
    print("5. CREATING WEIGHTED ENSEMBLE")
    print("=" * 100)
    
    weights = calculate_ensemble_weights(all_results, weight_metric='roc_auc')
    print("\nCalculated weights based on CV ROC-AUC:")
    for model_name, weight in weights.items():
        print(f"  {model_name}: {weight:.4f} ({weight*100:.2f}%)")
    
    print("\nGenerating ensemble predictions...")
    ensemble_proba, ensemble_pred = create_weighted_ensemble_predictions(
        trained_models, X_test, weights
    )
    
    print("Evaluating ensemble performance...")
    ensemble_results = evaluate_ensemble(y_test, ensemble_pred, ensemble_proba)
    
    # Print all results
    print_comparison_table(all_results, ensemble_results, weights)
    print_ensemble_details(ensemble_results, weights)
    print_feature_importance_comparison(all_results, top_n=10)
    print_feature_importance_consensus(all_results, top_n=10)
    
    # Save results
    save_results_to_csv(all_results, ensemble_results, weights)
    
    print("\n" + "=" * 100)
    print("AGGREGATE ANALYSIS COMPLETE!")
    print("=" * 100)
    
    return {
        'individual_results': all_results,
        'ensemble_results': ensemble_results,
        'weights': weights,
        'trained_models': trained_models
    }


def main():
    """Main function to run aggregate analysis"""
    parser = argparse.ArgumentParser(description='Run aggregate analysis with weighted ensemble')
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
