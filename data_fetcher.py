#!/usr/bin/env python3
"""
Data fetching and indicator calculation module
Fetches stock data and calculates technical indicators
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import time
from data_cache import DataCache


def fetch_stock_data(symbol, start_date="2018-01-01", end_date=None):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data (default: 2018-01-01)
        end_date: End date for data (default: today)
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        print(f"Fetching {symbol} data from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date, 
                          auto_adjust=False, repair=True, progress=False)
        
        if data.empty:
            print(f"Warning: No data returned for {symbol}")
            return None
        
        # Ensure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: Missing required columns for {symbol}")
            return None
        
        # Clean up index
        data.index = pd.to_datetime(data.index)
        if hasattr(data.index, 'date'):
            data.index = pd.to_datetime([d.date() if hasattr(d, 'date') else d for d in data.index])
        
        # Sort by date
        data = data.sort_index()
        
        print(f"Successfully fetched {len(data)} days of {symbol} data")
        return data
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    Already in 0-100 range
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_mfi(high, low, close, volume, period=14):
    """
    Calculate Money Flow Index (MFI)
    Already in 0-100 range
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_flow_sum / negative_flow_sum))
    
    return mfi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns MACD line, signal line, and histogram
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    Returns upper band, middle band (SMA), lower band, and %B
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    middle_band = sma
    
    # %B: (Price - Lower Band) / (Upper Band - Lower Band)
    # Normalize to 0-100 range
    percent_b = ((close - lower_band) / (upper_band - lower_band)) * 100
    
    return upper_band, middle_band, lower_band, percent_b


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator
    Already in 0-100 range (%K and %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Already in 0-100 range
    """
    # Calculate True Range - use numpy maximum to avoid DataFrame issues
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # Get max of the three using numpy
    tr = pd.Series(np.maximum.reduce([tr1.values, tr2.values, tr3.values]), index=close.index)
    
    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.clip(lower=0)  # Set negative values to 0
    minus_dm = minus_dm.clip(lower=0)  # Set negative values to 0
    
    # Calculate smoothed TR and DM
    atr = tr.rolling(window=period).mean()
    # Handle division by zero
    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)
    
    # Calculate ADX - handle division by zero
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)  # Avoid division by zero
    dx = 100 * abs(plus_di - minus_di) / di_sum_safe
    adx = dx.rolling(window=period).mean()
    
    # Ensure all are Series
    adx = pd.Series(adx, index=close.index, name='adx')
    plus_di = pd.Series(plus_di, index=close.index, name='plus_di')
    minus_di = pd.Series(minus_di, index=close.index, name='minus_di')
    
    return adx, plus_di, minus_di


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR)
    Will be normalized later
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_volume_indicators(close, volume):
    """
    Calculate volume-based indicators
    """
    # Volume SMA
    volume_sma = volume.rolling(window=20).mean()
    
    # Volume Ratio (current volume / average volume)
    volume_ratio = (volume / volume_sma) * 100  # Normalize to 0-100+ range
    
    # On Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    return volume_sma, volume_ratio, obv


def calculate_price_features(close, high, low):
    """
    Calculate additional price-based features
    """
    # Price change percentages
    pct_change_1d = close.pct_change(1) * 100
    pct_change_5d = close.pct_change(5) * 100
    pct_change_10d = close.pct_change(10) * 100
    
    # Moving averages
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()
    
    # Price relative to moving averages (normalized to 0-100+ range)
    price_to_sma20 = (close / sma_20) * 100
    price_to_sma50 = (close / sma_50) * 100
    price_to_sma200 = (close / sma_200) * 100
    
    # High-Low range percentage
    hl_range_pct = ((high - low) / close) * 100
    
    return {
        'pct_change_1d': pct_change_1d,
        'pct_change_5d': pct_change_5d,
        'pct_change_10d': pct_change_10d,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'price_to_sma20': price_to_sma20,
        'price_to_sma50': price_to_sma50,
        'price_to_sma200': price_to_sma200,
        'hl_range_pct': hl_range_pct
    }


def calculate_all_indicators(data):
    """
    Calculate all technical indicators for a stock
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all calculated indicators
    """
    df = data.copy()
    
    # Extract OHLCV - ensure they are Series
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    open_price = df['Open'].squeeze() if isinstance(df['Open'], pd.DataFrame) else df['Open']
    volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    
    # Ensure all are Series with same index
    if not isinstance(high, pd.Series):
        high = pd.Series(high, index=df.index)
    if not isinstance(low, pd.Series):
        low = pd.Series(low, index=df.index)
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=df.index)
    if not isinstance(volume, pd.Series):
        volume = pd.Series(volume, index=df.index)
    
    # Calculate indicators
    print("Calculating RSI...")
    df['rsi'] = calculate_rsi(close)
    
    print("Calculating MFI...")
    df['mfi'] = calculate_mfi(high, low, close, volume)
    
    print("Calculating MACD...")
    macd_line, signal_line, histogram = calculate_macd(close)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    print("Calculating Bollinger Bands...")
    bb_upper, bb_middle, bb_lower, bb_percent_b = calculate_bollinger_bands(close)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_percent_b'] = bb_percent_b
    
    print("Calculating Stochastic Oscillator...")
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    print("Calculating ADX...")
    adx, plus_di, minus_di = calculate_adx(high, low, close)
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    print("Calculating ATR...")
    df['atr'] = calculate_atr(high, low, close)
    
    print("Calculating Volume Indicators...")
    volume_sma, volume_ratio, obv = calculate_volume_indicators(close, volume)
    df['volume_sma'] = volume_sma
    df['volume_ratio'] = volume_ratio
    df['obv'] = obv
    
    print("Calculating Price Features...")
    price_features = calculate_price_features(close, high, low)
    for key, value in price_features.items():
        df[key] = value
    
    return df


def fetch_and_prepare_data(symbols, start_date="2018-01-01", end_date=None, delay=0.2, use_cache=True, force_refresh=False):
    """
    Fetch and prepare data for multiple symbols
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date for data
        end_date: End date for data
        delay: Delay between API calls (seconds)
        use_cache: Whether to use cached data if available (default: True)
        force_refresh: If True, ignore cache and fetch fresh data (default: False)
    
    Returns:
        Dictionary mapping symbols to DataFrames with indicators
    """
    all_data = {}
    cache = DataCache() if use_cache else None
    
    # Show cache info if using cache
    if cache and not force_refresh:
        cache_info = cache.get_cache_info()
        if cache_info['symbol_count'] > 0:
            print(f"\nCache status: {cache_info['symbol_count']} symbols cached")
            print(f"  Oldest update: {cache_info['oldest_update']}")
            print(f"  Newest update: {cache_info['newest_update']}")
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
        print("=" * 60)
        
        # Try to load from cache first
        cached_data = None
        if cache and not force_refresh:
            cached_data = cache.get_cached_data(symbol, start_date, end_date)
        
        if cached_data is not None:
            all_data[symbol] = cached_data
            print(f"✓ Using cached data for {symbol}")
            continue
        
        # Fetch stock data
        data = fetch_stock_data(symbol, start_date, end_date)
        
        if data is None:
            print(f"Skipping {symbol} due to data fetch error")
            continue
        
        # Calculate all indicators
        try:
            data_with_indicators = calculate_all_indicators(data)
            all_data[symbol] = data_with_indicators
            
            # Cache the processed data
            if cache:
                cache.cache_data(symbol, data_with_indicators, start_date, end_date)
            
            print(f"✓ Successfully processed {symbol}")
        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")
            continue
        
        # Delay to avoid overwhelming APIs
        if i < len(symbols) - 1:
            time.sleep(delay)
    
    return all_data

