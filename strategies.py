from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

# Constants for numerical stability
EPSILON = 1e-8  # Small value to prevent division by zero

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    position_type: 'PositionType' = None  # Will be set to PositionType.NONE by default
    take_profit: float = None
    stop_loss: float = None
    pnl: float = 0
    status: str = "open"  # "open", "closed", "tp_hit", "sl_hit"
    
    def __post_init__(self):
        if self.position_type is None:
            self.position_type = PositionType.NONE

class Signal(Enum):
    HOLD = 0
    LONG_ENTRY = 1
    LONG_EXIT = 2
    SHORT_ENTRY = -1
    SHORT_EXIT = -2

class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1

class BaseStrategy(ABC):
    def __init__(self, name: str, risk_reward_ratio: float = 2.0, atr_period: int = 14, trading_fee: float = 0.0):
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be a non-empty string")
        
        if risk_reward_ratio <= 0:
            raise ValueError(f"Risk-reward ratio must be positive, got: {risk_reward_ratio}")
        
        if atr_period <= 0:
            raise ValueError(f"ATR period must be positive, got: {atr_period}")
        
        if trading_fee < 0:
            raise ValueError(f"Trading fee must be non-negative, got: {trading_fee}")
        
        self.name = name
        self.signals = None
        self.current_position = PositionType.NONE  # No position, LONG, or SHORT
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_period = atr_period
        self.trading_fee = trading_fee  # Trading fee as a percentage (e.g., 0.001 for 0.1%)
        self.trades: List[Trade] = []
        self.active_trade: Trade = None
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range using exponential moving average.
        
        This method implements Welles Wilder's ATR using EMA for better responsiveness
        in fast-changing markets. The output is a Pandas Series aligned with the input data.
        
        Args:
            data (pd.DataFrame): OHLCV data with columns 'High', 'Low', 'Close'
            
        Returns:
            pd.Series: ATR values aligned with the input data index
            
        Raises:
            ValueError: If required columns are missing or atr_period is invalid
        """
        # Validate input data
        required_columns = ['High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate atr_period
        if self.atr_period <= 0:
            raise ValueError(f"ATR period must be positive, got: {self.atr_period}")
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # True Range is the maximum of the three components
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Use exponential moving average for ATR (more responsive than simple MA)
        # Handle NaN values from .shift() operation
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        
        return atr
    
    def calculate_trade_levels(self, entry_price: float, position_type: PositionType, atr: float) -> tuple:
        """
        Calculate take profit and stop loss levels based on ATR.
        
        This method implements a classic ATR-based volatility stop system:
        - Stop Loss: 1 ATR from entry price
        - Take Profit: R × ATR from entry price (where R is the risk_reward_ratio)
        
        Args:
            entry_price (float): Entry price of the trade
            position_type (PositionType): LONG or SHORT position type
            atr (float): Average True Range value for volatility measurement
            
        Returns:
            tuple: (take_profit, stop_loss) price levels
            
        Raises:
            ValueError: If position_type is invalid or ATR is negative
        """
        # Validate inputs
        if position_type not in [PositionType.LONG, PositionType.SHORT]:
            raise ValueError(f"Position type must be PositionType.LONG or PositionType.SHORT, got: {position_type}")
        
        if atr < 0:
            raise ValueError(f"ATR must be non-negative, got: {atr}")
        
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got: {entry_price}")
        
        # Handle NaN or None ATR values
        if pd.isna(atr) or atr is None or atr == 0:
            # Use a default ATR value (1% of entry price) if ATR is invalid
            atr = entry_price * 0.01
            print(f"Warning: Invalid ATR value ({atr}), using default 1% of entry price: {atr}")
        
        if position_type == PositionType.LONG:  # Long position
            stop_loss = entry_price - atr
            take_profit = entry_price + (atr * self.risk_reward_ratio)
        else:  # Short position
            stop_loss = entry_price + atr
            take_profit = entry_price - (atr * self.risk_reward_ratio)
        
        return take_profit, stop_loss
    
    def get_strategy_metrics(self, since_date: pd.Timestamp = None, recent_bars: set = None) -> Dict:
        """Calculate strategy performance metrics, optionally for trades closed since a given date or in a set of recent bars (timestamps)."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "total_pnl": 0,
                "geometric_mean_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "risk_reward_ratio": 0,
                "calmar_ratio": 0,
                "sortino_ratio": 0
            }
        
        closed_trades = [t for t in self.trades if t.status != "open"]
        if recent_bars is not None:
            closed_trades = [t for t in closed_trades if t.exit_date is not None and t.exit_date in recent_bars]
        elif since_date is not None:
            closed_trades = [t for t in closed_trades if t.exit_date is not None and t.exit_date >= since_date]
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "total_pnl": 0,
                "geometric_mean_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "risk_reward_ratio": 0,
                "calmar_ratio": 0,
                "sortino_ratio": 0
            }
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        # Basic metrics
        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_return = sum(t.pnl for t in closed_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate total PnL properly using compounding
        total_multiplier = 1.0
        for trade in closed_trades:
            total_multiplier *= (1 + trade.pnl)
        total_pnl = total_multiplier - 1.0
        
        # Geometric mean return (average return per trade)
        geometric_mean_return = (total_multiplier ** (1.0 / total_trades)) - 1.0 if total_trades > 0 else 0
        
        # Win/Loss metrics
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        largest_win = max(t.pnl for t in closed_trades) if closed_trades else 0
        largest_loss = min(t.pnl for t in closed_trades) if closed_trades else 0
        
        # Risk-reward ratio (avg_win / abs(avg_loss)) - this is actually the win/loss ratio
        # For true risk-reward ratio, we'd need to know the intended risk per trade
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        # Profit factor (total profit / total loss)
        total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf') if total_profit > 0 else 0
        
        # Consecutive wins/losses - fixed logic
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for trade in closed_trades:
            if trade.pnl > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0  # Reset loss streak
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_losses += 1
                current_consecutive_wins = 0  # Reset win streak
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        returns = [t.pnl for t in closed_trades]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe_ratio = mean_return / max(std_return, EPSILON)
        else:
            sharpe_ratio = 0
        
        # Calculate Sortino ratio (downside deviation)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns, ddof=1)
                sortino_ratio = mean_return / max(downside_deviation, EPSILON)
            else:
                sortino_ratio = float('inf') if mean_return > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative_return = 1.0
        for trade in closed_trades:
            cumulative_return *= (1 + trade.pnl)
            cumulative_returns.append(cumulative_return)
        
        max_drawdown = 0
        peak = 1.0
        for cr in cumulative_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar ratio (annualized return / max drawdown)
        # Calculate trading days between first and last trade for proper annualization
        if len(closed_trades) > 1:
            first_trade_date = min(t.exit_date for t in closed_trades if t.exit_date is not None)
            last_trade_date = max(t.exit_date for t in closed_trades if t.exit_date is not None)
            trading_days = (last_trade_date - first_trade_date).days
            if trading_days > 0:
                # Correct annualization: (total_multiplier ** (365 / trading_days)) - 1
                annualized_return = (total_multiplier ** (365 / trading_days)) - 1
            else:
                # Fallback: use geometric mean with daily assumption
                annualized_return = (1 + geometric_mean_return) ** 252 - 1
        else:
            # Fallback: use geometric mean with daily assumption
            annualized_return = (1 + geometric_mean_return) ** 252 - 1
        
        calmar_ratio = annualized_return / max(max_drawdown, EPSILON)
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_pnl": total_pnl,
            "geometric_mean_return": geometric_mean_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "consecutive_wins": max_consecutive_wins,
            "consecutive_losses": max_consecutive_losses,
            "risk_reward_ratio": win_loss_ratio,  # Renamed for clarity
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio
        }

    def calculate_pnl_with_fees(self, entry_price: float, exit_price: float, position_type: PositionType) -> float:
        """
        Calculate PnL including trading fees
        
        Args:
            entry_price (float): Entry price of the trade
            exit_price (float): Exit price of the trade
            position_type (PositionType): LONG or SHORT position type
            
        Returns:
            float: PnL as a percentage including fees
        """
        if position_type == PositionType.LONG:  # Long position
            # Gross PnL = (exit_price - entry_price) / entry_price
            gross_pnl = (exit_price - entry_price) / entry_price
            
            # Trading fees are a percentage of the trade value at each point
            # Entry fee: trading_fee × entry_price
            # Exit fee: trading_fee × exit_price
            # Total fees as percentage of entry price: (entry_fee + exit_fee) / entry_price
            entry_fee = self.trading_fee * entry_price
            exit_fee = self.trading_fee * exit_price
            total_fees_pct = (entry_fee + exit_fee) / entry_price
            
            net_pnl = gross_pnl - total_fees_pct
        else:  # Short position
            # Gross PnL = (entry_price - exit_price) / entry_price
            gross_pnl = (entry_price - exit_price) / entry_price
            
            # For short positions, fees work the same way
            # Entry fee: trading_fee × entry_price (when selling)
            # Exit fee: trading_fee × exit_price (when buying back)
            entry_fee = self.trading_fee * entry_price
            exit_fee = self.trading_fee * exit_price
            total_fees_pct = (entry_fee + exit_fee) / entry_price
            
            net_pnl = gross_pnl - total_fees_pct
        
        return net_pnl

    def get_cumulative_pnl_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cumulative PnL data for plotting performance over time.
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            DataFrame with columns: ['Date', 'Cumulative_PnL', 'Drawdown', 'Peak']
        """
        try:
            if not self.trades:
                # Return empty DataFrame with same structure
                return pd.DataFrame({
                    'Date': data.index,
                    'Cumulative_PnL': [1.0] * len(data),
                    'Drawdown': [0.0] * len(data),
                    'Peak': [1.0] * len(data)
                })
            
            # Get closed trades only
            closed_trades = [t for t in self.trades if t.status != "open" and t.exit_date is not None]
            
            if not closed_trades:
                return pd.DataFrame({
                    'Date': data.index,
                    'Cumulative_PnL': [1.0] * len(data),
                    'Drawdown': [0.0] * len(data),
                    'Peak': [1.0] * len(data)
                })
            
            # Sort trades by exit date
            closed_trades.sort(key=lambda x: x.exit_date)
            
            # Create cumulative PnL series
            cumulative_pnl = []
            dates = []
            current_pnl = 1.0  # Start with 100%
            
            # Initialize with starting point
            if data.index[0] < closed_trades[0].exit_date:
                cumulative_pnl.append(1.0)
                dates.append(data.index[0])
            
            # Add PnL points for each trade
            for trade in closed_trades:
                current_pnl *= (1 + trade.pnl)
                cumulative_pnl.append(current_pnl)
                dates.append(trade.exit_date)
            
            # Add final point if needed
            if data.index[-1] > closed_trades[-1].exit_date:
                cumulative_pnl.append(current_pnl)
                dates.append(data.index[-1])
            
            # Create DataFrame
            pnl_df = pd.DataFrame({
                'Date': dates,
                'Cumulative_PnL': cumulative_pnl
            })
            
            # Calculate drawdown
            pnl_df['Peak'] = pnl_df['Cumulative_PnL'].expanding().max()
            pnl_df['Drawdown'] = (pnl_df['Peak'] - pnl_df['Cumulative_PnL']) / pnl_df['Peak']
            
            # Handle duplicate dates by keeping the last value for each date
            pnl_df = pnl_df.drop_duplicates(subset=['Date'], keep='last')
            
            # Set index and reindex to match data frequency
            pnl_df = pnl_df.set_index('Date')
            
            # Ensure the index is unique before reindexing
            if not pnl_df.index.is_unique:
                # If there are still duplicates, keep the last occurrence
                pnl_df = pnl_df[~pnl_df.index.duplicated(keep='last')]
            
            # Reindex to match data frequency and interpolate
            full_index = data.index
            pnl_df = pnl_df.reindex(full_index, method='ffill')
            pnl_df = pnl_df.fillna(1.0)  # Fill any remaining NaNs with starting value
            
            # Reset index and ensure proper column naming
            pnl_df = pnl_df.reset_index()
            pnl_df = pnl_df.rename(columns={pnl_df.columns[0]: 'Date'})
            
            # Validate the final DataFrame
            required_columns = ['Date', 'Cumulative_PnL', 'Drawdown', 'Peak']
            if not all(col in pnl_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}, Got: {list(pnl_df.columns)}")
            
            return pnl_df
            
        except Exception as e:
            # Return a safe default DataFrame if anything goes wrong
            return pd.DataFrame({
                'Date': data.index,
                'Cumulative_PnL': [1.0] * len(data),
                'Drawdown': [0.0] * len(data),
                'Peak': [1.0] * len(data)
            })

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get detailed trade history for analysis.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            if trade.status != "open" and trade.exit_date is not None:
                trade_data.append({
                    'Entry_Date': trade.entry_date,
                    'Exit_Date': trade.exit_date,
                    'Entry_Price': trade.entry_price,
                    'Exit_Price': trade.exit_price,
                    'Position_Type': 'Long' if trade.position_type == PositionType.LONG else 'Short',
                    'PnL': trade.pnl,
                    'PnL_Pct': trade.pnl * 100,
                    'Status': trade.status,
                    'Take_Profit': trade.take_profit,
                    'Stop_Loss': trade.stop_loss
                })
        
        return pd.DataFrame(trade_data)

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
    
    def generate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate strategy indicators without modifying internal state.
        This method is safe to call multiple times for charting purposes.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with indicators added
        """
        # Default implementation - override in subclasses
        return data.copy()

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50, risk_reward_ratio: float = 2.0, trading_fee: float = 0.0):
        super().__init__("Moving Average Crossover", risk_reward_ratio, trading_fee=trading_fee)
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
        
        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Generate signals
        df['Signal'] = Signal.HOLD.value
        df['Position'] = 0
        df['Take_Profit'] = np.nan
        df['Stop_Loss'] = np.nan
        
        # Initialize position tracking
        current_position = PositionType.NONE
        
        for i in range(1, len(df)):
            prev_short = df['SMA_short'].iloc[i-1]
            prev_long = df['SMA_long'].iloc[i-1]
            curr_short = df['SMA_short'].iloc[i]
            curr_long = df['SMA_long'].iloc[i]
            current_price = df['Close'].iloc[i]
            current_atr = df['ATR'].iloc[i]
            
            # Skip iteration if any required values are NaN (early periods)
            if np.isnan(prev_short) or np.isnan(prev_long) or np.isnan(curr_short) or np.isnan(curr_long) or np.isnan(current_atr):
                continue  # Skip iteration
            
            # Check for crossover
            if prev_short <= prev_long and curr_short > curr_long:
                # Long signal - close any existing short position first
                if self.active_trade and self.active_trade.position_type == PositionType.SHORT:
                    # Exit existing short trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.SHORT
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE]:  # Enter long position
                    df.loc[df.index[i], 'Signal'] = Signal.LONG_ENTRY.value
                    current_position = PositionType.LONG
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.LONG, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.LONG,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
            elif prev_short >= prev_long and curr_short < curr_long:
                # Short signal - close any existing long position first
                if self.active_trade and self.active_trade.position_type == PositionType.LONG:
                    # Exit existing long trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.LONG
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE]:  # Enter short position
                    df.loc[df.index[i], 'Signal'] = Signal.SHORT_ENTRY.value
                    current_position = PositionType.SHORT
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.SHORT, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.SHORT,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
            
            # Check for take profit or stop loss
            if self.active_trade:
                if self.active_trade.position_type == PositionType.LONG:  # Long position
                    if current_price >= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price <= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                else:  # Short position
                    if current_price <= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price >= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
            
            # Update position
            df.loc[df.index[i], 'Position'] = current_position.value
            
            # Carry forward take profit and stop loss values if there's an active trade
            if self.active_trade:
                df.loc[df.index[i], 'Take_Profit'] = self.active_trade.take_profit
                df.loc[df.index[i], 'Stop_Loss'] = self.active_trade.stop_loss
        
        return df
    
    def generate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate moving average indicators without modifying internal state.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with indicators added
        """
        df = data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Add signal column (empty for indicators only)
        df['Signal'] = 0
        
        return df

class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, risk_reward_ratio: float = 2.0, trading_fee: float = 0.0):
        super().__init__("RSI Strategy", risk_reward_ratio, trading_fee=trading_fee)
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
        
        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Generate signals
        df['Signal'] = Signal.HOLD.value
        df['Position'] = 0
        df['Take_Profit'] = np.nan
        df['Stop_Loss'] = np.nan
        
        # Initialize position tracking
        current_position = PositionType.NONE
        
        for i in range(1, len(df)):
            prev_rsi = df['RSI'].iloc[i-1]
            curr_rsi = df['RSI'].iloc[i]
            current_price = df['Close'].iloc[i]
            current_atr = df['ATR'].iloc[i]
            
            # Skip iteration if any required values are NaN (early periods)
            if np.isnan(prev_rsi) or np.isnan(curr_rsi) or np.isnan(current_atr):
                continue  # Skip iteration
            
            # Check for oversold condition (potential long entry)
            if prev_rsi > self.oversold and curr_rsi <= self.oversold:
                # Long signal - close any existing short position first
                if self.active_trade and self.active_trade.position_type == PositionType.SHORT:
                    # Exit existing short trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.SHORT
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE, PositionType.SHORT]:  # Enter long position
                    df.loc[df.index[i], 'Signal'] = Signal.LONG_ENTRY.value
                    current_position = PositionType.LONG
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.LONG, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.LONG,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
            # Check for overbought condition (potential short entry)
            elif prev_rsi < self.overbought and curr_rsi >= self.overbought:
                # Short signal - close any existing long position first
                if self.active_trade and self.active_trade.position_type == PositionType.LONG:
                    # Exit existing long trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.LONG
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE, PositionType.LONG]:  # Enter short position
                    df.loc[df.index[i], 'Signal'] = Signal.SHORT_ENTRY.value
                    current_position = PositionType.SHORT
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.SHORT, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.SHORT,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
            
            # Check for take profit or stop loss
            if self.active_trade:
                if self.active_trade.position_type == PositionType.LONG:  # Long position
                    if current_price >= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price <= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                else:  # Short position
                    if current_price <= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price >= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
            
            # Update position
            df.loc[df.index[i], 'Position'] = current_position.value
            
            # Carry forward take profit and stop loss values if there's an active trade
            if self.active_trade:
                df.loc[df.index[i], 'Take_Profit'] = self.active_trade.take_profit
                df.loc[df.index[i], 'Stop_Loss'] = self.active_trade.stop_loss
        
        return df
    
    def generate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RSI indicators without modifying internal state.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with indicators added
        """
        df = data.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add signal column (empty for indicators only)
        df['Signal'] = 0
        
        return df

class DonchianChannelBreakout(BaseStrategy):
    def __init__(self, channel_period: int = 20, risk_reward_ratio: float = 2.0, trading_fee: float = 0.0):
        super().__init__("Donchian Channel Breakout", risk_reward_ratio, trading_fee=trading_fee)
        self.channel_period = channel_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Donchian Channel breakout
        Args:
            data (pd.DataFrame): OHLCV data
        Returns:
            pd.DataFrame: Data with signals added
        """
        df = data.copy()
        
        # Calculate Donchian Channel
        df['Donchian_High'] = df['High'].rolling(window=self.channel_period).max()
        df['Donchian_Low'] = df['Low'].rolling(window=self.channel_period).min()
        
        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Generate signals
        df['Signal'] = Signal.HOLD.value
        df['Position'] = 0
        df['Take_Profit'] = np.nan
        df['Stop_Loss'] = np.nan
        
        current_position = PositionType.NONE
        
        for i in range(self.channel_period, len(df)):
            current_price = df['Close'].iloc[i]
            current_atr = df['ATR'].iloc[i]
            donchian_high = df['Donchian_High'].iloc[i-1]  # Use previous bar for breakout
            donchian_low = df['Donchian_Low'].iloc[i-1]

            # Skip iteration if any required values are NaN (early periods)
            if np.isnan(current_atr) or np.isnan(donchian_high) or np.isnan(donchian_low):
                continue  # Skip iteration

            # Long breakout
            if current_price > donchian_high:
                # Long signal - close any existing short position first
                if self.active_trade and self.active_trade.position_type == PositionType.SHORT:
                    # Exit existing short trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.SHORT
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE, PositionType.SHORT]:
                    df.loc[df.index[i], 'Signal'] = Signal.LONG_ENTRY.value
                    current_position = PositionType.LONG
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.LONG, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.LONG,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
            # Short breakout
            elif current_price < donchian_low:
                # Short signal - close any existing long position first
                if self.active_trade and self.active_trade.position_type == PositionType.LONG:
                    # Exit existing long trade at current price
                    self.active_trade.exit_date = df.index[i]
                    self.active_trade.exit_price = current_price
                    self.active_trade.pnl = self.calculate_pnl_with_fees(
                        self.active_trade.entry_price, current_price, PositionType.LONG
                    )
                    self.active_trade.status = "reversed"
                    self.trades.append(self.active_trade)
                    self.active_trade = None
                    current_position = PositionType.NONE
                
                if current_position in [PositionType.NONE, PositionType.LONG]:
                    df.loc[df.index[i], 'Signal'] = Signal.SHORT_ENTRY.value
                    current_position = PositionType.SHORT
                    take_profit, stop_loss = self.calculate_trade_levels(current_price, PositionType.SHORT, current_atr)
                    df.loc[df.index[i], 'Take_Profit'] = take_profit
                    df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                    self.active_trade = Trade(
                        entry_date=df.index[i],
                        entry_price=current_price,
                        position_type=PositionType.SHORT,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )

            # Check for take profit or stop loss
            if self.active_trade:
                if self.active_trade.position_type == PositionType.LONG:  # Long
                    if current_price >= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price <= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.LONG_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.LONG
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                else:  # Short
                    if current_price <= self.active_trade.take_profit:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "tp_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
                    elif current_price >= self.active_trade.stop_loss:
                        df.loc[df.index[i], 'Signal'] = Signal.SHORT_EXIT.value
                        self.active_trade.exit_date = df.index[i]
                        self.active_trade.exit_price = current_price
                        self.active_trade.pnl = self.calculate_pnl_with_fees(
                            self.active_trade.entry_price, current_price, PositionType.SHORT
                        )
                        self.active_trade.status = "sl_hit"
                        self.trades.append(self.active_trade)
                        self.active_trade = None
                        current_position = PositionType.NONE
            
            df.loc[df.index[i], 'Position'] = current_position.value
            
            # Carry forward take profit and stop loss values if there's an active trade
            if self.active_trade:
                df.loc[df.index[i], 'Take_Profit'] = self.active_trade.take_profit
                df.loc[df.index[i], 'Stop_Loss'] = self.active_trade.stop_loss
        
        return df 