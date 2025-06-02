from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import Enum

class Signal(Enum):
    HOLD = 0
    LONG_ENTRY = 1
    LONG_EXIT = 2
    SHORT_ENTRY = -1
    SHORT_EXIT = -2

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.signals = None
        self.current_position = 0  # 0: no position, 1: long, -1: short
    
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
        df['Position'] = 0
        
        # Initialize position tracking
        current_position = 0
        
        for i in range(1, len(df)):
            prev_short = df['SMA_short'].iloc[i-1]
            prev_long = df['SMA_long'].iloc[i-1]
            curr_short = df['SMA_short'].iloc[i]
            curr_long = df['SMA_long'].iloc[i]
            
            # Check for crossover
            if prev_short <= prev_long and curr_short > curr_long:
                if current_position <= 0:  # Only enter if not already long
                    df.loc[df.index[i], 'Signal'] = Signal.LONG_ENTRY.value
                    current_position = 1
            elif prev_short >= prev_long and curr_short < curr_long:
                if current_position >= 0:  # Only enter if not already short
                    df.loc[df.index[i], 'Signal'] = Signal.SHORT_ENTRY.value
                    current_position = -1
            
            # Update position
            df.loc[df.index[i], 'Position'] = current_position
        
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
        df['Position'] = 0
        
        # Initialize position tracking
        current_position = 0
        
        for i in range(1, len(df)):
            prev_rsi = df['RSI'].iloc[i-1]
            curr_rsi = df['RSI'].iloc[i]
            
            # Check for oversold condition (potential long entry)
            if prev_rsi > self.oversold and curr_rsi <= self.oversold:
                if current_position <= 0:  # Only enter if not already long
                    df.loc[df.index[i], 'Signal'] = Signal.LONG_ENTRY.value
                    current_position = 1
            # Check for overbought condition (potential short entry)
            elif prev_rsi < self.overbought and curr_rsi >= self.overbought:
                if current_position >= 0:  # Only enter if not already short
                    df.loc[df.index[i], 'Signal'] = Signal.SHORT_ENTRY.value
                    current_position = -1
            
            # Update position
            df.loc[df.index[i], 'Position'] = current_position
        
        return df 