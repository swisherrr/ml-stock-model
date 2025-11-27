#!/usr/bin/env python3
"""
Model training module
Trains XGBoost classifier with time-series cross-validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, classification_report, 
                            confusion_matrix)
import pickle
import os
from datetime import datetime


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        y: Target labels
    
    Returns:
        Dictionary with class weights
    """
    from collections import Counter
    counter = Counter(y)
    total = len(y)
    
    # Calculate weights inversely proportional to class frequency
    weight_0 = total / (2 * counter[0]) if counter[0] > 0 else 1.0
    weight_1 = total / (2 * counter[1]) if counter[1] > 0 else 1.0
    
    return {0: weight_0, 1: weight_1}


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
    
    # Final train/test split (temporal)
    test_start_idx = int(n_samples * (1 - test_size))
    train_indices = np.arange(0, test_start_idx)
    test_indices = np.arange(test_start_idx, n_samples)
    
    # Cross-validation splits (temporal)
    cv_splits = []
    train_size = len(train_indices)
    
    for i in range(n_splits):
        # Each fold uses more data for training
        fold_train_size = int(train_size * (i + 1) / n_splits)
        fold_train_idx = train_indices[:fold_train_size]
        fold_val_start = fold_train_size
        fold_val_end = min(fold_train_size + int(train_size / n_splits), train_size)
        fold_val_idx = train_indices[fold_val_start:fold_val_end]
        
        if len(fold_val_idx) > 0:
            cv_splits.append((fold_train_idx, fold_val_idx))
    
    return cv_splits, (train_indices, test_indices)


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, 
                        class_weights=None, n_estimators=100, 
                        max_depth=6, learning_rate=0.1, random_state=42):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        class_weights: Dictionary with class weights
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
    
    Returns:
        Trained XGBoost model
    """
    # Calculate class weights if not provided
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    print(f"Class weights: {class_weights}")
    
    # Prepare scale_pos_weight (ratio of negative to positive samples)
    scale_pos_weight = class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0
    
    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist'
    )
    
    # Train model
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)
    
    return model


def evaluate_model(model, X, y, set_name="Test"):
    """
    Evaluate model and return metrics
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        set_name: Name of the dataset (for printing)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{set_name} Set Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
    print(f"    False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def cross_validate(X, y, n_splits=5, n_estimators=100, max_depth=6, 
                   learning_rate=0.1, random_state=42):
    """
    Perform time-series cross-validation
    
    Args:
        X: Features DataFrame
        y: Target Series
        n_splits: Number of CV folds
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
    
    Returns:
        Dictionary with CV results
    """
    print("Performing time-series cross-validation...")
    
    # Create time-series splits
    cv_splits, _ = time_series_split(X, y, n_splits=n_splits)
    
    cv_results = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}/{len(cv_splits)}")
        print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = train_xgboost_model(
            X_train_fold, y_train_fold,
            class_weights=calculate_class_weights(y_train_fold),
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val_fold, y_val_fold, 
                                set_name=f"CV Fold {fold + 1}")
        
        # Store results
        cv_results['precision'].append(metrics['precision'])
        cv_results['recall'].append(metrics['recall'])
        cv_results['f1_score'].append(metrics['f1_score'])
        cv_results['roc_auc'].append(metrics['roc_auc'])
    
    # Print CV summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary:")
    print(f"{'='*60}")
    for metric in ['precision', 'recall', 'f1_score', 'roc_auc']:
        values = cv_results[metric]
        print(f"  {metric.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    
    return cv_results


def generate_per_stock_predictions(model, X_test, y_test, metadata):
    """
    Generate predictions grouped by stock
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        metadata: DataFrame with ticker, date, close_price for each sample
    
    Returns:
        DataFrame with per-stock predictions (latest prediction for each stock)
    """
    if metadata is None or 'ticker' not in metadata.columns:
        return None
    
    # Reset index for alignment
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test.values)
    metadata_reset = metadata.reset_index(drop=True) if metadata is not None else None
    
    # Make predictions
    probabilities = model.predict_proba(X_test_reset)[:, 1]
    predictions = model.predict(X_test_reset)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'ticker': metadata_reset['ticker'] if metadata_reset is not None else None,
        'date': metadata_reset['date'] if metadata_reset is not None else None,
        'close_price': metadata_reset['close_price'] if metadata_reset is not None else None,
        'probability_up': probabilities,
        'prediction': predictions,
        'actual': y_test_reset.values if hasattr(y_test_reset, 'values') else y_test_reset
    })
    
    # Group by ticker and get the latest prediction for each stock
    latest_predictions = results_df.sort_values('date').groupby('ticker').last().reset_index()
    latest_predictions = latest_predictions[['ticker', 'date', 'close_price', 'probability_up', 'prediction']]
    latest_predictions = latest_predictions.sort_values('probability_up', ascending=False)
    
    return latest_predictions


def train_and_evaluate(X, y, feature_names, test_size=0.2, n_splits=5,
                       n_estimators=100, max_depth=6, learning_rate=0.1,
                       random_state=42, save_model=True, model_dir='models', metadata=None):
    """
    Complete training and evaluation pipeline
    
    Args:
        X: Features DataFrame
        y: Target Series
        feature_names: List of feature names
        test_size: Proportion of data for test set
        n_splits: Number of CV folds
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        save_model: Whether to save the trained model
        model_dir: Directory to save model
        metadata: DataFrame with ticker, date, close_price for tracking individual stocks
    
    Returns:
        Trained model and evaluation results (including per-stock predictions if metadata provided)
    """
    print(f"\n{'='*60}")
    print("Training XGBoost Model")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Time-series split
    _, (train_idx, test_idx) = time_series_split(X, y, n_splits=n_splits, 
                                                  test_size=test_size)
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = pd.Series(y.iloc[train_idx].values) if isinstance(y, pd.Series) else pd.Series(y.iloc[train_idx])
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = pd.Series(y.iloc[test_idx].values) if isinstance(y, pd.Series) else pd.Series(y.iloc[test_idx])
    
    # Split metadata if provided - ensure same length as test set
    metadata_test = None
    if metadata is not None and len(metadata) == len(X):
        metadata_test = metadata.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Cross-validation
    cv_results = cross_validate(X_train, y_train, n_splits=n_splits,
                                n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate, random_state=random_state)
    
    # Train final model on all training data
    print(f"\n{'='*60}")
    print("Training Final Model")
    print(f"{'='*60}")
    
    final_model = train_xgboost_model(
        X_train, y_train,
        class_weights=calculate_class_weights(y_train),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Final Test Set Evaluation")
    print(f"{'='*60}")
    test_metrics = evaluate_model(final_model, X_test, y_test, set_name="Test")
    
    # Generate per-stock predictions if metadata is available
    per_stock_predictions = None
    if metadata_test is not None:
        per_stock_predictions = generate_per_stock_predictions(final_model, X_test, y_test, metadata_test)
    
    # Feature importance
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances")
    print(f"{'='*60}")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    # Save model
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'xgboost_model_{timestamp}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'feature_names': feature_names,
                'cv_results': cv_results,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance
            }, f)
        
        print(f"\nModel saved to: {model_path}")
    
    return final_model, {
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'per_stock_predictions': per_stock_predictions
    }

