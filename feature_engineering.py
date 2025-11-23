#!/usr/bin/env python3
"""
Feature engineering and normalization module
Normalizes indicators to 0-100 range where applicable
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_to_0_100(series, min_val=None, max_val=None):
    """
    Normalize a series to 0-100 range
    
    Args:
        series: pandas Series to normalize
        min_val: Minimum value for normalization (if None, use series min)
        max_val: Maximum value for normalization (if None, use series max)
    
    Returns:
        Normalized Series in 0-100 range
    """
    # Ensure we have a Series
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    if min_val is None:
        min_val = series.min()
    if max_val is None:
        max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(50.0, index=series.index)  # Return middle value if no range
    
    normalized = ((series - min_val) / (max_val - min_val)) * 100
    result = normalized.clip(0, 100)  # Clip to ensure 0-100 range
    
    # Ensure result is Series
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=series.index)
    
    return result


def prepare_features(df):
    """
    Prepare and normalize all features for ML model
    
    Args:
        df: DataFrame with all indicators
    
    Returns:
        DataFrame with normalized features ready for ML
    """
    features_df = pd.DataFrame(index=df.index)
    
    # Indicators already in 0-100 range (keep as is, but ensure they're in range)
    already_normalized = ['rsi', 'mfi', 'stoch_k', 'stoch_d', 'adx', 
                         'plus_di', 'minus_di', 'bb_percent_b']
    
    for col in already_normalized:
        if col in df.columns:
            # Clip to 0-100 range - ensure Series
            col_data = df[col].squeeze() if isinstance(df[col], pd.DataFrame) else df[col]
            if not isinstance(col_data, pd.Series):
                col_data = pd.Series(col_data, index=df.index)
            features_df[col] = col_data.clip(0, 100)
    
    # MACD components - normalize to 0-100
    if 'macd' in df.columns:
        macd = df['macd'].squeeze() if isinstance(df['macd'], pd.DataFrame) else df['macd']
        features_df['macd_norm'] = normalize_to_0_100(macd)
    if 'macd_signal' in df.columns:
        macd_signal = df['macd_signal'].squeeze() if isinstance(df['macd_signal'], pd.DataFrame) else df['macd_signal']
        features_df['macd_signal_norm'] = normalize_to_0_100(macd_signal)
    if 'macd_histogram' in df.columns:
        macd_hist = df['macd_histogram'].squeeze() if isinstance(df['macd_histogram'], pd.DataFrame) else df['macd_histogram']
        features_df['macd_histogram_norm'] = normalize_to_0_100(macd_hist)
    
    # Bollinger Bands - normalize upper, middle, lower relative to price
    if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'Close']):
        # Normalize BB bands relative to current price
        # Ensure we're working with Series
        bb_upper = df['bb_upper'].squeeze() if isinstance(df['bb_upper'], pd.DataFrame) else df['bb_upper']
        bb_lower = df['bb_lower'].squeeze() if isinstance(df['bb_lower'], pd.DataFrame) else df['bb_lower']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        bb_range = bb_upper - bb_lower
        # Handle division by zero
        bb_range = bb_range.replace(0, np.nan)
        
        bb_upper_norm = ((bb_upper - close) / bb_range) * 100 + 50
        bb_lower_norm = ((close - bb_lower) / bb_range) * 100 + 50
        
        # Ensure Series and clip
        if not isinstance(bb_upper_norm, pd.Series):
            bb_upper_norm = pd.Series(bb_upper_norm, index=df.index)
        if not isinstance(bb_lower_norm, pd.Series):
            bb_lower_norm = pd.Series(bb_lower_norm, index=df.index)
        
        features_df['bb_upper_norm'] = bb_upper_norm.clip(0, 100)
        features_df['bb_lower_norm'] = bb_lower_norm.clip(0, 100)
    
    # ATR - normalize relative to price (as percentage)
    if 'atr' in df.columns and 'Close' in df.columns:
        # Ensure we're working with Series
        atr = df['atr'].squeeze() if isinstance(df['atr'], pd.DataFrame) else df['atr']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        atr_pct = (atr / close) * 100
        # Ensure result is Series
        if not isinstance(atr_pct, pd.Series):
            atr_pct = pd.Series(atr_pct, index=df.index)
        features_df['atr_norm'] = normalize_to_0_100(atr_pct)
    
    # Volume indicators
    if 'volume_ratio' in df.columns:
        # Volume ratio is already a percentage, but may exceed 100
        # Normalize to 0-100 by capping at reasonable max (e.g., 500%)
        features_df['volume_ratio_norm'] = (df['volume_ratio'] / 5).clip(0, 100)
    
    if 'obv' in df.columns:
        features_df['obv_norm'] = normalize_to_0_100(df['obv'])
    
    # Price change percentages - normalize to 0-100
    # Assuming typical range of -10% to +10% for daily changes
    for col in ['pct_change_1d', 'pct_change_5d', 'pct_change_10d']:
        if col in df.columns:
            # Shift from -10 to +10 range to 0-100 range
            features_df[f'{col}_norm'] = ((df[col] + 10) / 20) * 100
            features_df[f'{col}_norm'] = features_df[f'{col}_norm'].clip(0, 100)
    
    # Price to SMA ratios - already percentages, normalize to 0-100
    for col in ['price_to_sma20', 'price_to_sma50', 'price_to_sma200']:
        if col in df.columns:
            # Assuming range of 50% to 150% of SMA
            features_df[f'{col}_norm'] = ((df[col] - 50) / 100) * 100
            features_df[f'{col}_norm'] = features_df[f'{col}_norm'].clip(0, 100)
    
    # High-Low range percentage - already a percentage
    if 'hl_range_pct' in df.columns:
        # Assuming range of 0% to 10%
        features_df['hl_range_pct_norm'] = (df['hl_range_pct'] / 0.1).clip(0, 100)
    
    # Add some lag features (previous day values)
    for col in ['rsi', 'mfi', 'stoch_k', 'stoch_d']:
        if col in df.columns:
            features_df[f'{col}_lag1'] = df[col].shift(1).clip(0, 100)
    
    # Add momentum features
    if 'rsi' in df.columns:
        features_df['rsi_momentum'] = df['rsi'].diff()
        features_df['rsi_momentum'] = normalize_to_0_100(features_df['rsi_momentum'], 
                                                         min_val=-20, max_val=20)
    
    if 'mfi' in df.columns:
        features_df['mfi_momentum'] = df['mfi'].diff()
        features_df['mfi_momentum'] = normalize_to_0_100(features_df['mfi_momentum'],
                                                         min_val=-20, max_val=20)
    
    return features_df


def combine_all_features(all_data_dict):
    """
    Combine features from all stocks into a single DataFrame
    
    Args:
        all_data_dict: Dictionary mapping symbols to DataFrames with indicators
    
    Returns:
        Combined DataFrame with features and ticker information
    """
    all_features = []
    
    for symbol, df in all_data_dict.items():
        # Prepare features (features_df has same index as df)
        features_df = prepare_features(df)
        
        # Add ticker symbol
        features_df['ticker'] = symbol
        
        # Add date (use index as date)
        features_df['date'] = features_df.index
        
        # Add close price for label calculation
        features_df['close_price'] = df['Close']
        
        all_features.append(features_df)
    
    # Combine all DataFrames (preserve index for now)
    combined_df = pd.concat(all_features, axis=0)
    
    # Sort by date and ticker
    combined_df = combined_df.sort_values(['date', 'ticker'])
    
    return combined_df


def get_feature_columns(df):
    """
    Get list of feature columns (excluding metadata columns)
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of feature column names
    """
    exclude_cols = ['ticker', 'date', 'close_price', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

