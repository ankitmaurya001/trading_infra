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
    
    def __init__(self, credentials: Dict[str, Any], exchange='NSE', token_cache_file='access_token.txt'):
        """
        Initialize Kite Connect data fetcher
        
        Args:
            credentials: Dictionary containing:
                - api_key: Your Kite Connect API key
                - api_secret: Your Kite Connect API secret
                - username (optional): Username for login
                - password (optional): Password for login
                - totp_key (optional): TOTP key for 2FA
                - access_token (optional): Access token if already authenticated
            exchange: Exchange name (default: 'NSE')
            token_cache_file: Path to file where access token will be cached (default: 'access_token.txt')
        """
        self.kite = KiteConnect(api_key=credentials["api_key"])
        self.credentials = credentials
        self.instrument_tokens = {}
        self.exchange = exchange
        self.token_cache_file = token_cache_file
        
        # If access_token is provided in credentials, use it directly (skip cache)
        if "access_token" in credentials and credentials["access_token"]:
            self.kite.set_access_token(credentials["access_token"])
            print("[INFO] Using provided access_token (skipping cache check)")
            self._load_instruments()
        else:
            # Try to load cached access token first
            cached_token = self._load_cached_token()
            if cached_token:
                self.kite.set_access_token(cached_token)
                # Verify token is still valid
                if self._is_token_valid():
                    print(f"[INFO] Using cached access token from {self.token_cache_file}")
                    self._load_instruments()
                else:
                    print(f"[INFO] Cached token expired or invalid, will authenticate...")
                    self.kite._access_token = None  # Clear invalid token


    def authenticate(self):
        """
        Authenticate with Kite Connect.
        If access token is already set and valid, this method does nothing.
        Otherwise, performs full authentication and caches the token.
        """
        # Check if we already have a valid token
        if hasattr(self.kite, '_access_token') and self.kite._access_token:
            if self._is_token_valid():
                print("[INFO] Already authenticated with valid token, skipping authentication...")
                return
        
        session = requests.Session()
        # Step 1: Get login URL
        response = session.get(self.kite.login_url())
        # Step 2: Login form
        login_payload = {
            "user_id": self.credentials["username"],
            "password": self.credentials["password"],
        }
        login_response = session.post("https://kite.zerodha.com/api/login", login_payload)
        
        # Check login response and extract request_id
        try:
            login_json = login_response.json()
            print(f"[INFO] login_response.status_code = {login_response.status_code}")
            print(f"[INFO] login_response.json() = {login_json}")
        except Exception as e:
            print(f"[ERROR] Could not parse login response: {e}")
            print(f"[ERROR] Response text: {login_response.text}")
            raise ValueError(f"Login failed: Could not parse response - {e}")
        
        if login_response.status_code != 200:
            error_json = login_json if 'login_json' in locals() else {}
            error_msg = error_json.get('message', login_response.text)
            
            # Check for account lock or CAPTCHA
            if error_json.get('data', {}).get('captcha'):
                raise ValueError(
                    f"⚠️  CAPTCHA Required / Account Locked!\n"
                    f"   Status: {login_response.status_code}\n"
                    f"   Message: {error_msg}\n\n"
                    f"   This usually happens when:\n"
                    f"   1. Account is locked due to repeated 2FA failures\n"
                    f"   2. Too many failed login attempts\n\n"
                    f"   Solutions:\n"
                    f"   1. Unlock account: Login to Kite Web and unlock if locked\n"
                    f"   2. Verify TOTP: Run 'python test_totp.py' to test TOTP generation\n"
                    f"   3. Wait: Account may auto-unlock after some time\n"
                    f"   4. Contact Zerodha: If account remains locked\n"
                )
            
            raise ValueError(f"Login failed with status {login_response.status_code}: {error_msg}")
        
        # Extract request_id - try different possible structures
        request_id = None
        if "data" in login_json and isinstance(login_json["data"], dict):
            request_id = login_json["data"].get("request_id")
        elif "request_id" in login_json:
            request_id = login_json["request_id"]
        else:
            # Try to find request_id anywhere in nested structure
            def find_request_id(obj):
                if isinstance(obj, dict):
                    if "request_id" in obj:
                        return obj["request_id"]
                    for value in obj.values():
                        result = find_request_id(value)
                        if result is not None:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_request_id(item)
                        if result is not None:
                            return result
                return None
            
            request_id = find_request_id(login_json)
        
        if request_id is None:
            raise ValueError(f"Could not find 'request_id' in login response. Response structure: {login_json}")
        
        print(f"[INFO] request_id = {request_id}")
        
        # Step 3: TOTP 2FA
        totp_value = otp.get_totp(self.credentials["totp_key"])
        print(f"[INFO] Generated TOTP: {totp_value}")
        
        totp_payload = {
            "user_id": self.credentials["username"],
            "request_id": request_id,
            "twofa_value": totp_value,
            "twofa_type": "totp",
            "skip_session": True,
        }
        totp_response = session.post("https://kite.zerodha.com/api/twofa", totp_payload)
        print(f"[INFO] totp_response.status_code = {totp_response.status_code}")
        
        # Check TOTP response
        if totp_response.status_code != 200:
            try:
                totp_error = totp_response.json()
                print(f"[INFO] totp_response.json() = {totp_error}")
                if "Invalid" in str(totp_error) or "invalid" in str(totp_error).lower():
                    raise ValueError(
                        f"⚠️  TOTP Authentication Failed!\n"
                        f"   Status: {totp_response.status_code}\n"
                        f"   Error: {totp_error}\n\n"
                        f"   Possible causes:\n"
                        f"   1. TOTP key is incorrect or expired\n"
                        f"   2. TOTP is out of sync (time drift)\n"
                        f"   3. Account is locked\n\n"
                        f"   Solutions:\n"
                        f"   1. Test TOTP: Run 'python test_totp.py' to verify TOTP generation\n"
                        f"   2. Verify TOTP key in config matches Kite Web → Settings → API → TOTP\n"
                        f"   3. Regenerate TOTP in Kite if needed\n"
                        f"   4. Unlock account if locked\n"
                    )
            except:
                pass
            raise ValueError(f"TOTP authentication failed: {totp_response.status_code} - {totp_response.text}")

        request_token = None
        # Step 4: Extract request token
        try:
            response = session.get(self.kite.login_url())
            parse_result = urlparse(response.url)
            query_parms = parse_qs(parse_result.query)
        except Exception as e:
            import re
            pattern = r"request_token=[A-Za-z0-9]+"
            matches = re.findall(pattern, str(e))
            if len(matches) == 0:
                raise ValueError(f"[ERROR] Could not extract request_token from exception: {e}")
            query_parms = parse_qs(matches[0])
        if "request_token" not in query_parms:
            raise ValueError("[ERROR] request_token not found in query parameters")
        request_token = query_parms["request_token"][0]
        print(f"[INFO] request_token = {request_token}")

        if request_token == None:
            raise ValueError("[ERROR] Request token not found")
        data = self.kite.generate_session(request_token, api_secret=self.credentials["api_secret"])
        #print(data)
        access_token = data["access_token"]
        self.kite.set_access_token(access_token)
        
        # Save access token to cache file
        self._save_token_to_cache(access_token)
        print(f"[INFO] Access token saved to {self.token_cache_file}")
        
        profile = self.fetch_profile()
        self._load_instruments()
    
    def _save_token_to_cache(self, access_token: str):
        """
        Save access token to cache file.
        
        Args:
            access_token: The access token to save
        """
        try:
            import os
            # Save to file in the same directory as the script
            cache_path = os.path.join(os.path.dirname(__file__), self.token_cache_file)
            with open(cache_path, 'w') as f:
                f.write(access_token)
            # Set restrictive permissions (read/write for owner only)
            os.chmod(cache_path, 0o600)
        except Exception as e:
            logging.warning(f"Could not save access token to cache: {e}")
    
    def _load_cached_token(self) -> Optional[str]:
        """
        Load access token from cache file if it exists.
        
        Returns:
            Access token string if found, None otherwise
        """
        try:
            import os
            cache_path = os.path.join(os.path.dirname(__file__), self.token_cache_file)
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    token = f.read().strip()
                    if token:
                        return token
        except Exception as e:
            logging.debug(f"Could not load cached token: {e}")
        return None
    
    def _is_token_valid(self) -> bool:
        """
        Check if the current access token is valid by making a simple API call.
        
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Try to fetch profile - this is a lightweight API call
            self.kite.profile()
            return True
        except Exception as e:
            # Token is invalid or expired
            logging.debug(f"Token validation failed: {e}")
            return False
        
    def fetch_profile(self):
        """Fetch profile from Kite Connect"""
        profile = None
        try:
            profile = self.kite.profile()
            print(f"[INFO] profile = {profile}")
            return profile
        except Exception as e:
            logging.error(f"Error fetching profile: {e}")
            return None
        print(f"[INFO] profile = {profile}")
        return profile
        
       
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
            error_msg = str(e)
            # Log the full error for debugging
            logging.error(f"Error fetching historical data for {symbol}: {error_msg}")
            
            # Re-raise authentication errors so they can be handled upstream
            if any(keyword in error_msg.lower() for keyword in ['api_key', 'access_token', 'authentication', 'unauthorized', 'token']):
                logging.warning(f"⚠️  Authentication error detected: {error_msg}")
                raise  # Re-raise to allow upstream handling
            
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
            
            # Try exact match first
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    self.instrument_tokens[symbol] = instrument['instrument_token']
                    return instrument['instrument_token']
            
            # Try partial match (symbol contains search term)
            matching_symbols = [inst['tradingsymbol'] for inst in instruments if symbol.upper() in inst['tradingsymbol'].upper() or inst['tradingsymbol'].upper() in symbol.upper()]
            
            if matching_symbols:
                # Use the first match (could be improved with better matching logic)
                matched_symbol = matching_symbols[0]
                for instrument in instruments:
                    if instrument['tradingsymbol'] == matched_symbol:
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

    