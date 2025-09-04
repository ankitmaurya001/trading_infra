import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import pytz
from datetime import datetime, timedelta
import logging
import requests
import re
from urllib.parse import urlparse, parse_qs
import onetimepass as otp

# Kite Connect imports
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logging.warning("KiteConnect not available. Install with: pip install kiteconnect")

class DataFetcher:
    def __init__(self):
        self.data = None
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str, key_point_multiplier: float = 2.0, interval: str = "1d", to_ist: bool = True) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance at a specified interval
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            key_point_multiplier (float): Multiplier for average daily return to identify key points
            interval (str): Data interval (e.g., '1d', '5m', '15m', '1h', etc.)
            to_ist (bool): If True, convert the DataFrame index to IST (Asia/Kolkata) timezone
        Returns:
            pd.DataFrame: Cleaned OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            self.data = stock.history(start=start_date, end=end_date, interval=interval)
            df = self.clean_data(key_point_multiplier)
            if not df.empty and to_ist:
                # Convert index to IST
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
            return df
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

class KiteDataFetcher:
    """
    Data fetcher for Zerodha Kite Connect API
    Provides real-time and historical data for Indian markets
    """
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Initialize Kite Connect data fetcher
        
        Args:
            api_key (str): Your Kite Connect API key
            api_secret (str): Your Kite Connect API secret
            access_token (str, optional): Access token if already authenticated
        """
        
            
        self.kite = KiteConnect(api_key=credentials["api_key"])
        self.credentials = credentials
        self.instrument_tokens = {}


    def authenticate(self):
        session = requests.Session()
        # Step 1: Get login URL
        response = session.get(self.kite.login_url())
        # Step 2: Login form
        login_payload = {
            "user_id": self.credentials["username"],
            "password": self.credentials["password"],
        }
        login_response = session.post("https://kite.zerodha.com/api/login", login_payload)
        # Step 3: TOTP 2FA
        totp_payload = {
            "user_id": self.credentials["username"],
            "request_id": login_response.json()["data"]["request_id"],
            "twofa_value": otp.get_totp(self.credentials["totp_key"]),
            "twofa_type": "totp",
            "skip_session": True,
        }
        totp_response = session.post("https://kite.zerodha.com/api/twofa", totp_payload)
        print(f"[INFO] totp_respone = {totp_response}")

        request_token = None
        # Step 4: Extract request token
        try:
            response = session.get(self.kite.login_url())
            parse_result = urlparse(response.url)
            query_parms = parse_qs(parse_result.query)
        except Exception as e:
            import re
            pattern = r"request_token=[A-Za-z0-9]+"
            query_parms = parse_qs(re.findall(pattern, str(e))[0])
        request_token = query_parms["request_token"][0]
        print(f"[INFO] request_token = {request_token}")

        if request_token == None:
            raise ValueError("[ERROR] Request token not found")
        data = self.kite.generate_session(request_token, api_secret=self.credentials["api_secret"])
        #print(data)
        self.kite.set_access_token(data["access_token"])
        self._load_instruments()
        
       
    def _load_instruments(self):
        """Load instrument tokens for common stocks"""
        try:
            # Load NSE instruments
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] in ['TATAMOTORS', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK']:
                    self.instrument_tokens[instrument['tradingsymbol']] = instrument['instrument_token']
        except Exception as e:
            logging.warning(f"Could not load instruments: {e}")
    
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                            interval: str = "15minute", continuous: bool = False) -> pd.DataFrame:
        """
        Fetch historical data from Kite Connect
        
        Args:
            symbol (str): Trading symbol (e.g., 'TATAMOTORS')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('minute', '5minute', '15minute', '30minute', '60minute', 'day')
            continuous (bool): If True, returns continuous data
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Get instrument token for the symbol
            instrument_token = self._get_instrument_token(symbol)
            print(f"[INFO] instrument_token: {instrument_token}")
            if not instrument_token:
                raise ValueError(f"Instrument token not found for {symbol}")
            
            # Convert dates to datetime objects
            start_dt = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            end_dt = (datetime.now() +timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval=interval,
                continuous=continuous
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Clean and format the data
            df = self._clean_kite_data(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """
        Get instrument token for a given symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            int: Instrument token
        """
        # Check if we have it cached
        if symbol in self.instrument_tokens:
            return self.instrument_tokens[symbol]
        
        # Search for the instrument
        try:
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    self.instrument_tokens[symbol] = instrument['instrument_token']
                    return instrument['instrument_token']
        except Exception as e:
            logging.error(f"Error searching for instrument {symbol}: {e}")
        
        return None
    
    def _clean_kite_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format Kite Connect data
        
        Args:
            df (pd.DataFrame): Raw data from Kite
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df
        
        # Rename columns to match standard format
        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Set Date as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Handle timezone - convert to IST if needed
        if df.index.tz is None:
            # If no timezone info, localize to IST
            df.index = df.index.tz_localize('Asia/Kolkata')
        else:
            # If already has timezone, convert to IST
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Sort by date
        df = df.sort_index()
        
        # Remove any missing values
        df = df.dropna()
        
        return df
    
    def fetch_tatamotors_data(self, start_date: str, end_date: str, interval: str = "15minute") -> pd.DataFrame:
        """
        Convenience method to fetch TATAMOTORS data
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: TATAMOTORS historical data
        """
        return self.fetch_historical_data('TATAMOTORS', start_date, end_date, interval)
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict: Real-time quote data
        """
        try:
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                raise ValueError(f"Instrument token not found for {symbol}")
            
            quote = self.kite.quote(f"NSE:{symbol}")
            return quote
        except Exception as e:
            logging.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get Last Traded Price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Last traded price
        """
        try:
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                return None
            
            ltp_data = self.kite.ltp(f"NSE:{symbol}")
            return ltp_data[f"NSE:{symbol}"]["last_price"]
        except Exception as e:
            logging.error(f"Error fetching LTP for {symbol}: {e}")
            return None 

    