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

# Binance imports
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("Binance client not available. Install with: pip install python-binance")

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
    
    def __init__(self, credentials: Dict[str, Any], exchange='NSE'):
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
        self.exchange = exchange


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
            instruments = self.kite.instruments(self.exchange)
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
            # start_dt = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            # end_dt = (datetime.now() +timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
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
            instruments = self.kite.instruments(self.exchange)
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    self.instrument_tokens[symbol] = instrument['instrument_token']
                    return instrument['instrument_token']
        except Exception as e:
            logging.error(f"Error searching for instrument {symbol}: {e}")
        
        return None
    
    def _clean_kite_data(self, df: pd.DataFrame, key_point_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Clean and format Kite Connect data with additional features like DataFetcher
        
        Args:
            df (pd.DataFrame): Raw data from Kite
            key_point_multiplier (float): Multiplier for average daily return to identify key points
            
        Returns:
            pd.DataFrame: Cleaned data with additional features
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
        
        # Add additional features like DataFetcher.clean_data()
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate average daily return
        df['Avg_Daily_Return'] = df['Returns'].rolling(window=20).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Calculate key points (where price moved more than multiplier * average daily return)
        df['Key_Point'] = abs(df['Returns']) > (key_point_multiplier * abs(df['Avg_Daily_Return']))
        
        return df
    
    def get_key_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get the key points where price moved significantly
        
        Args:
            data (pd.DataFrame): Data with Key_Point column
            
        Returns:
            pd.DataFrame: Dataframe containing only key points
        """
        if data.empty or 'Key_Point' not in data.columns:
            return pd.DataFrame()
            
        return data[data['Key_Point'] == True]
    
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
            
            quote = self.kite.quote(f"{self.exchange}:{symbol}")
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
            
            ltp_data = self.kite.ltp(f"{self.exchange}:{symbol}")
            return ltp_data[f"{self.exchange}:{symbol}"]["last_price"]
        except Exception as e:
            logging.error(f"Error fetching LTP for {symbol}: {e}")
            return None 


class BinanceDataFetcher:
    """
    Data fetcher for Binance API
    Provides historical data for cryptocurrency markets
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize Binance data fetcher
        
        Args:
            api_key (str): Your Binance API key (optional for public data)
            api_secret (str): Your Binance API secret (optional for public data)
            testnet (bool): If True, use Binance testnet
        """
        if not BINANCE_AVAILABLE:
            raise ImportError("Binance client not available. Install with: pip install python-binance")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize Binance client
        if api_key and api_secret:
            self.client = Client(api_key, api_secret, testnet=testnet)
        else:
            # Public client for historical data (no API key needed)
            self.client = Client(testnet=testnet)
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                            interval: str = "1m") -> pd.DataFrame:
        """
        Fetch historical data from Binance
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            start_date (str): Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            end_date (str): End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            interval (str): Data interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Parse datetime strings - support both date-only and full datetime formats
            try:
                # Try full datetime format first
                start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Fall back to date-only format (defaults to 00:00:00)
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            try:
                # Try full datetime format first
                end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Fall back to date-only format (defaults to 00:00:00)
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Map interval to Binance format
            interval_mapping = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '6h': Client.KLINE_INTERVAL_6HOUR,
                '8h': Client.KLINE_INTERVAL_8HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '3d': Client.KLINE_INTERVAL_3DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK,
                '1M': Client.KLINE_INTERVAL_1MONTH
            }
            
            binance_interval = interval_mapping.get(interval, Client.KLINE_INTERVAL_1MINUTE)
            
            # Fetch historical klines - use proper format for Binance API
            # Convert to milliseconds for Binance API
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=binance_interval,
                start_str=start_ms,
                end_str=end_ms
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            
            # Clean and format the data
            df = self._clean_binance_data(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _clean_binance_data(self, df: pd.DataFrame, key_point_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Clean and format Binance data with additional features like other fetchers
        
        Args:
            df (pd.DataFrame): Raw data from Binance
            key_point_multiplier (float): Multiplier for average daily return to identify key points
            
        Returns:
            pd.DataFrame: Cleaned data with additional features
        """
        if df.empty:
            return df
        
        # Convert numeric columns to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                          'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime and set as index
        df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df.set_index('Date')
        
        # Convert to IST timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Keep only essential OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Sort by date
        df = df.sort_index()
        
        # Remove any missing values
        df = df.dropna()
        
        # Add additional features like other fetchers
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate average daily return
        df['Avg_Daily_Return'] = df['Returns'].rolling(window=20).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Calculate key points (where price moved more than multiplier * average daily return)
        df['Key_Point'] = abs(df['Returns']) > (key_point_multiplier * abs(df['Avg_Daily_Return']))
        
        return df
    
    def get_key_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get the key points where price moved significantly
        
        Args:
            data (pd.DataFrame): Data with Key_Point column
            
        Returns:
            pd.DataFrame: Dataframe containing only key points
        """
        if data.empty or 'Key_Point' not in data.columns:
            return pd.DataFrame()
            
        return data[data['Key_Point'] == True]
    
    def fetch_btc_data(self, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """
        Convenience method to fetch BTC/USDT data
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            end_date (str): End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: BTC/USDT historical data
        """
        return self.fetch_historical_data('BTCUSDT', start_date, end_date, interval)
    
    def fetch_eth_data(self, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """
        Convenience method to fetch ETH/USDT data
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            end_date (str): End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: ETH/USDT historical data
        """
        return self.fetch_historical_data('ETHUSDT', start_date, end_date, interval)
    
    def fetch_latest_btc_data(self, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
        """
        Convenience method to fetch latest BTC/USDT data
        
        Args:
            interval (str): Data interval
            limit (int): Number of recent klines to fetch
            
        Returns:
            pd.DataFrame: Latest BTC/USDT data
        """
        return self.fetch_latest_data('BTCUSDT', interval, limit)
    
    def fetch_latest_eth_data(self, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
        """
        Convenience method to fetch latest ETH/USDT data
        
        Args:
            interval (str): Data interval
            limit (int): Number of recent klines to fetch
            
        Returns:
            pd.DataFrame: Latest ETH/USDT data
        """
        return self.fetch_latest_data('ETHUSDT', interval, limit)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information from Binance
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict: Symbol information
        """
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
            return {}
        except Exception as e:
            logging.error(f"Error fetching symbol info for {symbol}: {e}")
            return {}
    
    def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker price change statistics
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict: 24hr ticker data
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            logging.error(f"Error fetching 24hr ticker for {symbol}: {e}")
            return {}
    
    def fetch_latest_data(self, symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch the most recent data for a symbol
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            interval (str): Data interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            limit (int): Number of recent klines to fetch (max 1000)
            
        Returns:
            pd.DataFrame: Latest OHLCV data
        """
        try:
            # Map interval to Binance format
            interval_mapping = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '6h': Client.KLINE_INTERVAL_6HOUR,
                '8h': Client.KLINE_INTERVAL_8HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '3d': Client.KLINE_INTERVAL_3DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK,
                '1M': Client.KLINE_INTERVAL_1MONTH
            }
            
            binance_interval = interval_mapping.get(interval, Client.KLINE_INTERVAL_1MINUTE)
            
            # Fetch latest klines
            klines = self.client.get_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            
            # Clean and format the data
            df = self._clean_binance_data(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching latest data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return float(ticker.get('lastPrice', 0))
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None

    