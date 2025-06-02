import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class DataFetcher:
    def __init__(self):
        self.data = None
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str, key_point_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            key_point_multiplier (float): Multiplier for average daily return to identify key points
            
        Returns:
            pd.DataFrame: Cleaned OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            self.data = stock.history(start=start_date, end=end_date)
            return self.clean_data(key_point_multiplier)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, key_point_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Clean the fetched data by:
        1. Removing missing values
        2. Calculating daily returns
        3. Adding technical indicators
        
        Args:
            key_point_multiplier (float): Multiplier for average daily return to identify key points
            
        Returns:
            pd.DataFrame: Cleaned data with additional features
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        # Remove missing values
        df = self.data.dropna()
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate average daily return
        df['Avg_Daily_Return'] = df['Returns'].rolling(window=20).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Calculate key points (where price moved more than multiplier * average daily return)
        df['Key_Point'] = abs(df['Returns']) > (key_point_multiplier * abs(df['Avg_Daily_Return']))
        
        return df
    
    def get_key_points(self) -> pd.DataFrame:
        """
        Get the key points where price moved significantly
        
        Returns:
            pd.DataFrame: Dataframe containing only key points
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        return self.data[self.data['Key_Point'] == True] 