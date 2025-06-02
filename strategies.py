from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import Enum

class Signal(Enum):
    HOLD = 0
    BUY = 1
    SELL = -1

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals added
        """
        pass

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on moving average crossover
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals added
        """
        df = data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        df['Signal'] = Signal.HOLD.value
        
        # Buy signal: short MA crosses above long MA
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = Signal.BUY.value
        
        # Sell signal: short MA crosses below long MA
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = Signal.SELL.value
        
        # Only generate signals when both MAs are available
        df.loc[df['SMA_short'].isna() | df['SMA_long'].isna(), 'Signal'] = Signal.HOLD.value
        
        return df

class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI Strategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on RSI
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals added
        """
        df = data.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['Signal'] = Signal.HOLD.value
        
        # Buy signal: RSI crosses below oversold
        df.loc[df['RSI'] < self.oversold, 'Signal'] = Signal.BUY.value
        
        # Sell signal: RSI crosses above overbought
        df.loc[df['RSI'] > self.overbought, 'Signal'] = Signal.SELL.value
        
        return df 