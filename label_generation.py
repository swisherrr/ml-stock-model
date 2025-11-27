#!/usr/bin/env python3
"""
Label generation module
Creates binary labels for 5-day ahead price movement prediction
"""

import pandas as pd
import numpy as np


def create_labels(df, days_ahead=5):
    """
    Create binary labels: 1 if stock closes higher N days later, 0 otherwise
    
    Args:
        df: DataFrame with close_price column and date index
        days_ahead: Number of days ahead to predict (default: 5)
    
    Returns:
        Series with binary labels (1 = higher, 0 = lower or same)
    """
    labels = pd.Series(index=df.index, dtype=float)
    
    # Group by ticker to handle each stock separately
    if 'ticker' in df.columns:
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            ticker_data = ticker_data.sort_values('date' if 'date' in ticker_data.columns else ticker_data.index)
            
            # Calculate future close prices
            future_closes = ticker_data['close_price'].shift(-days_ahead)
            current_closes = ticker_data['close_price']
            
            # Create labels: 1 if future close > current close, 0 otherwise
            ticker_labels = (future_closes > current_closes).astype(int)
            
            # Assign labels back using original index
            labels.loc[ticker_mask] = ticker_labels.values
    else:
        # If no ticker column, assume single stock
        df_sorted = df.sort_index()
        future_closes = df_sorted['close_price'].shift(-days_ahead)
        current_closes = df_sorted['close_price']
        labels = (future_closes > current_closes).astype(int)
    
    return labels


def align_features_and_labels(features_df, labels):
    """
    Align features and labels, removing rows with NaN labels (no future data)
    
    Args:
        features_df: DataFrame with features
        labels: Series with labels
    
    Returns:
        Tuple of (aligned_features, aligned_labels) with NaN rows removed
    """
    # Combine features and labels
    combined = features_df.copy()
    combined['target'] = labels
    
    # Remove rows where target is NaN (no future data available)
    combined = combined.dropna(subset=['target'])
    
    # Separate features and labels
    aligned_features = combined.drop(columns=['target'])
    aligned_labels = combined['target'].astype(int)
    
    return aligned_features, aligned_labels


def prepare_ml_data(all_data_dict, days_ahead=5):
    """
    Prepare complete ML dataset with features and labels
    
    Args:
        all_data_dict: Dictionary mapping symbols to DataFrames with indicators
        days_ahead: Number of days ahead to predict (default: 5)
    
    Returns:
        Tuple of (features_df, labels_series, feature_columns, metadata_df)
        where metadata_df contains ticker, date, and close_price for tracking
    """
    import feature_engineering
    combine_all_features = feature_engineering.combine_all_features
    get_feature_columns = feature_engineering.get_feature_columns
    
    # Combine all features
    print("Combining features from all stocks...")
    combined_df = combine_all_features(all_data_dict)
    
    # Create labels
    print(f"Creating labels for {days_ahead}-day ahead prediction...")
    labels = create_labels(combined_df, days_ahead=days_ahead)
    
    # Align features and labels
    print("Aligning features and labels...")
    features_df, aligned_labels = align_features_and_labels(combined_df, labels)
    
    # Extract metadata (ticker, date, close_price) before selecting only features
    metadata_cols = ['ticker', 'date', 'close_price']
    metadata = features_df[metadata_cols].copy() if all(col in features_df.columns for col in metadata_cols) else None
    
    # Get feature columns (excluding metadata)
    feature_cols = get_feature_columns(features_df)
    
    # Select only feature columns for ML
    ml_features = features_df[feature_cols].copy()
    
    # Remove rows with any NaN values in features
    valid_mask = ~ml_features.isna().any(axis=1)
    ml_features = ml_features[valid_mask]
    aligned_labels = aligned_labels[valid_mask]
    
    # Apply same mask to metadata - keep the same index as ml_features
    if metadata is not None:
        metadata = metadata.loc[valid_mask].copy()
        # Reset index to match ml_features index (0, 1, 2, ...)
        metadata = metadata.reset_index(drop=True)
    
    # Reset indices for alignment
    ml_features = ml_features.reset_index(drop=True)
    if isinstance(aligned_labels, pd.Series):
        aligned_labels = aligned_labels.reset_index(drop=True)
    else:
        aligned_labels = pd.Series(aligned_labels.values)
    
    print(f"Final dataset: {len(ml_features)} samples, {len(feature_cols)} features")
    print(f"Label distribution: {aligned_labels.value_counts().to_dict()}")
    
    return ml_features, aligned_labels, feature_cols, metadata

