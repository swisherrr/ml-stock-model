#!/usr/bin/env python3
"""
Data caching module
Stores processed stock data with indicators in SQLite database
"""

import os
import sqlite3
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, Optional, List


class DataCache:
    """
    Manages caching of processed stock data in SQLite database
    """
    
    def __init__(self, db_path: str = "data_cache.db"):
        """
        Initialize the cache database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing processed data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_data (
                symbol TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                data BLOB NOT NULL,
                PRIMARY KEY (symbol, start_date)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol 
            ON processed_data(symbol)
        """)
        
        conn.commit()
        conn.close()
    
    def get_cached_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data (optional, for validation)
        
        Returns:
            DataFrame with processed data if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data, end_date, last_updated
            FROM processed_data
            WHERE symbol = ? AND start_date = ?
        """, (symbol, start_date))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return None
        
        data_blob, cached_end_date, last_updated = result
        
        # If end_date is specified, check if cached data is up to date
        if end_date is not None:
            # If cached end_date is before requested end_date, data might be stale
            if cached_end_date and cached_end_date < end_date:
                return None
        
        # Deserialize the DataFrame
        try:
            df = pickle.loads(data_blob)
            print(f"✓ Loaded cached data for {symbol} (last updated: {last_updated})")
            return df
        except Exception as e:
            print(f"Error deserializing cached data for {symbol}: {e}")
            return None
    
    def cache_data(self, symbol: str, data: pd.DataFrame, start_date: str, end_date: Optional[str] = None):
        """
        Store processed data in cache
        
        Args:
            symbol: Stock ticker symbol
            data: DataFrame with processed data (including indicators)
            start_date: Start date for data
            end_date: End date for data (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the DataFrame
        data_blob = pickle.dumps(data)
        last_updated = datetime.now().isoformat()
        
        # Insert or replace existing data
        cursor.execute("""
            INSERT OR REPLACE INTO processed_data 
            (symbol, last_updated, start_date, end_date, data)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, last_updated, start_date, end_date, data_blob))
        
        conn.commit()
        conn.close()
        print(f"✓ Cached data for {symbol}")
    
    def get_all_cached_symbols(self, start_date: str) -> List[str]:
        """
        Get list of all symbols that have cached data for a given start_date
        
        Args:
            start_date: Start date to check
        
        Returns:
            List of symbol strings
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT symbol
            FROM processed_data
            WHERE start_date = ?
        """, (start_date,))
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return symbols
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data
        
        Args:
            symbol: If provided, clear only this symbol's cache. 
                   If None, clear all cached data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("DELETE FROM processed_data WHERE symbol = ?", (symbol,))
            print(f"Cleared cache for {symbol}")
        else:
            cursor.execute("DELETE FROM processed_data")
            print("Cleared all cached data")
        
        conn.commit()
        conn.close()
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cached data
        
        Returns:
            Dictionary with cache statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT symbol) as symbol_count,
                COUNT(*) as total_records,
                MIN(last_updated) as oldest_update,
                MAX(last_updated) as newest_update
            FROM processed_data
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return {
                'symbol_count': result[0],
                'total_records': result[1],
                'oldest_update': result[2],
                'newest_update': result[3]
            }
        else:
            return {
                'symbol_count': 0,
                'total_records': 0,
                'oldest_update': None,
                'newest_update': None
            }

