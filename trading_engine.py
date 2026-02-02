#!/usr/bin/env python3
"""
Trading Engine - Handles trade execution, position management, and performance tracking
Separates trading logic from the live trading simulator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import os
import json
import logging
import pytz
from strategies import Signal

from strategies import BaseStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Trading engine that handles:
    - Trade execution and management
    - Position tracking
    - Performance calculation
    - Logging and reporting
    """
    
    def __init__(self, initial_balance: float = 10000, max_leverage: float = 10.0, max_loss_percent: float = 2.0, 
                 atr_buffer_percent: float = 0.0, margin_buffer_percent: float = 20.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_leverage = max_leverage
        self.max_loss_percent = max_loss_percent
        self.atr_buffer_percent = atr_buffer_percent  # Buffer to widen SL/TP (e.g., 20 = 20% wider)
        self.margin_buffer_percent = margin_buffer_percent  # Buffer for margin requirement check (e.g., 20 = 20% extra margin required)
        self.trade_history = []
        self.active_trades = []
        self.session_id = None
        self.symbol = None
        
        # Optional external broker for real execution
        self.broker = None
        self.use_broker = False

        # Risk controls
        self.risk_limits = {
            'max_order_notional': None,
            'max_daily_notional': None,
            'max_open_orders': None,
            'enable_kill_switch': False
        }
        self.traded_notional_today = 0.0
        self.trading_day = None
        self.kill_switch_triggered = False
        
        # Logging setup
        self.trade_log_file = None
        self.decision_log_file = None
        self.data_log_file = None
        
        # Performance tracking
        self.total_pnl = 0
        self.unrealized_pnl = 0
        
    def setup_logging(self, session_id: str, symbol: str, session_folder: str = None):
        """
        Setup logging files for the trading session.
        
        Args:
            session_id: Unique session identifier
            symbol: Trading symbol
            session_folder: Path to session-specific folder (optional)
        """
        self.session_id = session_id
        self.symbol = symbol
        
        # Use session folder if provided, otherwise use default logs directory
        if session_folder:
            self.session_folder = session_folder
            os.makedirs(session_folder, exist_ok=True)
        else:
            self.session_folder = "logs"
            os.makedirs(self.session_folder, exist_ok=True)
        
        # Setup log files in session folder
        self.trade_log_file = os.path.join(self.session_folder, "trades.csv")
        self.decision_log_file = os.path.join(self.session_folder, "decisions.csv")
        self.data_log_file = os.path.join(self.session_folder, "market_data.csv")
        
        # Initialize trade log
        trade_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'action', 'price', 'quantity', 
            'leverage', 'position_size', 'atr', 'balance', 'pnl', 'trade_id', 'status', 'reject_reason'
        ])
        trade_log_df.to_csv(self.trade_log_file, index=False)
        
        # Initialize decision log
        decision_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'signal', 'signal_name', 'current_price',
            'current_balance', 'active_trades_count', 'position_type', 'take_profit', 'stop_loss', 'trade_status'
        ])
        decision_log_df.to_csv(self.decision_log_file, index=False)
        
        print(f"📁 Logging setup complete:")
        print(f"   Trade Log: {self.trade_log_file}")
        print(f"   Decision Log: {self.decision_log_file}")
        print(f"   Data Log: {self.data_log_file}")

    def set_risk_limits(self, limits: Dict):
        """Configure risk limits for live/testnet execution."""
        if not limits:
            return
        self.risk_limits['max_order_notional'] = limits.get('max_order_notional')
        self.risk_limits['max_daily_notional'] = limits.get('max_daily_notional')
        self.risk_limits['max_open_orders'] = limits.get('max_open_orders')
        self.risk_limits['enable_kill_switch'] = limits.get('enable_kill_switch', False)

    def _reset_daily_counters_if_needed(self, current_time: datetime):
        day = current_time.date()
        if self.trading_day != day:
            self.trading_day = day
            self.traded_notional_today = 0.0

    def _check_risk_and_update(self, expected_notional: float, current_time: datetime) -> Tuple[bool, str]:
        self._reset_daily_counters_if_needed(current_time)
        if self.kill_switch_triggered and self.risk_limits.get('enable_kill_switch'):
            return False, 'kill_switch_triggered'
        max_order = self.risk_limits.get('max_order_notional')
        if max_order is not None and expected_notional > max_order:
            if self.risk_limits.get('enable_kill_switch'):
                self.kill_switch_triggered = True
            return False, f'order_notional_exceeds_limit ({expected_notional} > {max_order})'
        max_daily = self.risk_limits.get('max_daily_notional')
        if max_daily is not None and (self.traded_notional_today + expected_notional) > max_daily:
            if self.risk_limits.get('enable_kill_switch'):
                self.kill_switch_triggered = True
            return False, f'daily_notional_exceeds_limit ({self.traded_notional_today + expected_notional} > {max_daily})'
        max_open = self.risk_limits.get('max_open_orders')
        if max_open is not None and len(self.active_trades) >= max_open:
            return False, f'max_open_orders_reached ({len(self.active_trades)} >= {max_open})'
        # Passed checks; update day notional pre-commit
        self.traded_notional_today += expected_notional
        return True, 'ok'
    
    def _refresh_balance_from_broker(self):
        """
        Refresh current_balance from broker to account for funds added mid-session.
        
        This ensures that:
        1. Newly added funds are considered for position sizing
        2. Balance is always in sync with actual broker balance
        
        Only updates if broker balance (minus margin already blocked) is higher than internal balance.
        """
        if not self.broker or not hasattr(self.broker, 'check_margins'):
            return
        
        try:
            margins = self.broker.check_margins()
            available_margin = margins.get('available', 0.0)
            utilised_margin = margins.get('utilised', 0.0)
            
            # For single ledger (Kite), also check equity utilised
            # Total account value = available + utilised
            total_broker_balance = available_margin + utilised_margin
            
            # If we have open positions, the utilised margin is blocked for those trades
            # The available_margin is what we can use for NEW trades
            # Our internal current_balance should reflect what's available for new positions
            
            # Calculate our expected internal balance based on active trades
            margin_blocked_internally = sum(
                trade.get('margin_used', 0) 
                for trade in self.active_trades 
                if trade.get('status') == 'open'
            )
            
            # If broker's available is higher than our internal current_balance,
            # funds were added - update our balance
            if available_margin > self.current_balance:
                old_balance = self.current_balance
                self.current_balance = available_margin
                print(f"💰 Balance refreshed from broker: ₹{old_balance:,.2f} → ₹{self.current_balance:,.2f} (funds added)")
                logger.info(f"Balance refreshed: {old_balance:.2f} → {self.current_balance:.2f}")
            
            # Also update initial_balance if total account value increased
            # (This helps with PnL calculations when funds are added)
            if total_broker_balance > self.initial_balance:
                self.initial_balance = total_broker_balance
                logger.info(f"Initial balance updated to reflect added funds: {self.initial_balance:.2f}")
                
        except Exception as e:
            logger.warning(f"Could not refresh balance from broker: {e}")
            # Don't fail the trade, just use existing balance
    
    def execute_trade(self, strategy: BaseStrategy, action: str, price: float, timestamp: datetime, data: pd.DataFrame = None) -> Dict:
        """
        Execute a new trade with leverage-based position sizing.
        
        Args:
            strategy: Strategy instance that generated the signal
            action: Trade action ('BUY' or 'SELL')
            price: Execution price
            timestamp: Trade timestamp
            data: Market data for ATR calculation
            
        Returns:
            Dictionary containing trade details
        """
        # Determine position type based on action
        from strategies import PositionType
        if action == 'BUY':
            position_type = PositionType.LONG
        elif action == 'SELL':
            position_type = PositionType.SHORT
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate ATR for leverage-based position sizing
        if data is not None and not data.empty:
            atr_series = strategy.calculate_atr(data)
            if not atr_series.empty:
                atr_value = atr_series.iloc[-1]
                # Check if ATR is NaN or invalid
                if pd.isna(atr_value) or atr_value <= 0:
                    current_atr = price * 0.01
                    print(f"⚠️  Warning: ATR is NaN or invalid ({atr_value}), using fallback 1% of price: ${current_atr:.5f}")
                else:
                    current_atr = float(atr_value)
            else:
                current_atr = price * 0.01
                print(f"⚠️  Warning: ATR series is empty, using fallback 1% of price: ${current_atr:.5f}")
        else:
            # Fallback to 1% of price if no data available
            current_atr = price * 0.01
            print(f"⚠️  Warning: No market data available for ATR calculation, using fallback 1% of price: ${current_atr:.5f}")
        
        # Refresh balance from broker before position sizing
        # This ensures newly added funds are considered for position sizing
        self._refresh_balance_from_broker()
        
        # Check if this is a Kite broker (for lot-based trading)
        is_kite_broker = hasattr(self.broker, 'place_gtt_order') if self.broker else False
        
        # Check if this is an MCX commodity (even without broker, for mock mode)
        # MCX commodities typically have "MCX" in exchange or symbol patterns like "NATGASMINI", "GOLDM", etc.
        is_mcx_commodity = False
        if hasattr(self, 'symbol') and self.symbol:
            # Check if symbol matches MCX commodity patterns
            mcx_patterns = ['NATGASMINI', 'GOLDM', 'SILVERM', 'CRUDEOILM', 'SILVERMIC', 'GOLDGUINEA']
            is_mcx_commodity = any(pattern in self.symbol.upper() for pattern in mcx_patterns)
        if hasattr(self, 'exchange') and self.exchange:
            is_mcx_commodity = is_mcx_commodity or (self.exchange.upper() == 'MCX')
        
        print(f"🔍 Debug: is_kite_broker = {is_kite_broker}, broker = {type(self.broker).__name__ if self.broker else None}, is_mcx_commodity = {is_mcx_commodity}")
        
        # Calculate leverage-based position size
        leverage, position_size, quantity = strategy.calculate_leverage_position_size(
            entry_price=price,
            position_type=position_type,
            atr=current_atr,
            available_balance=self.current_balance,
            max_leverage=self.max_leverage,
            max_loss_percent=self.max_loss_percent
        )
        
        # For commodity futures (Kite broker OR MCX commodities), quantity should be 1 (number of lots)
        # The calculated quantity is in base units, but we trade in lots
        original_quantity = quantity
        if is_kite_broker or is_mcx_commodity:
            quantity = 1  # 1 lot for commodity futures
            print(f"📦 Overriding quantity to 1 lot for commodity futures (original calculated: {original_quantity:.4f})")
        
        # Risk checks
        expected_notional = position_size
        ok, reason = self._check_risk_and_update(expected_notional, timestamp)
        if not ok:
            print(f"⛔ Trade rejected by risk controls: {reason}")
            rejection = {
                'id': len(self.trade_history) + 1,
                'strategy': strategy.name,
                'action': action,
                'entry_price': price,
                'entry_time': timestamp,
                'quantity': 0.0,
                'leverage': 0.0,
                'position_size': 0.0,
                'atr': current_atr,
                'status': 'rejected',
                'pnl': 0,
                'reject_reason': reason
            }
            self.trade_history.append(rejection)
            self._log_trade(rejection, price, timestamp)
            return rejection
        
        # Margin check for Kite broker (lot-based trading)
        # Check margin even in virtual mode to validate trades would be possible
        actual_lot_margin = None  # Will store actual margin for Kite trades
        if self.broker and hasattr(self.broker, 'get_order_margins'):
            try:
                # Use Kite's order_margins API to get ACTUAL margin requirement
                # This gives the real margin needed, not an estimate
                transaction_type = 'BUY' if position_type.name == 'LONG' else 'SELL'
                
                order_margins = self.broker.get_order_margins(
                    symbol=self.symbol,
                    transaction_type=transaction_type,
                    quantity=1,  # 1 lot for commodities
                    price=price,
                    order_type='MARKET'
                )
                
                actual_lot_margin = order_margins.get('total', 0.0)
                
                if actual_lot_margin == 0.0:
                    print(f"⚠️  Could not get actual margin requirement. Proceeding with trade...")
                    actual_lot_margin = None
                else:
                    # Check margin with buffer
                    margins = self.broker.check_margins()
                    available_margin = margins.get('available', 0.0)
                    required_with_buffer = actual_lot_margin * (1 + self.margin_buffer_percent / 100)
                    
                    if available_margin < required_with_buffer:
                        reason = f"Insufficient margin: {available_margin:.2f} < {required_with_buffer:.2f} (required: {actual_lot_margin:.2f})"
                        print(f"⛔ Trade rejected: {reason}")
                        rejection = {
                            'id': len(self.trade_history) + 1,
                            'strategy': strategy.name,
                            'action': action,
                            'entry_price': price,
                            'entry_time': timestamp,
                            'quantity': 0.0,
                            'leverage': 0.0,
                            'position_size': 0.0,
                            'atr': current_atr,
                            'status': 'rejected',
                            'pnl': 0,
                            'reject_reason': reason
                        }
                        self.trade_history.append(rejection)
                        self._log_trade(rejection, price, timestamp)
                        return rejection
                    else:
                        print(f"✅ Margin check passed: {available_margin:.2f} >= {required_with_buffer:.2f} (actual margin: {actual_lot_margin:.2f}, buffer: {self.margin_buffer_percent}%)")
            except Exception as e:
                print(f"⚠️  Margin check failed: {e}. Proceeding with trade...")
                import traceback
                traceback.print_exc()

        # Calculate stop loss and take profit levels (ATR-based)
        take_profit, stop_loss = strategy.calculate_trade_levels(price, position_type, current_atr)
        
        # Apply ATR buffer if configured (widens SL/TP to avoid getting stopped out by noise)
        # e.g., atr_buffer_percent=20 means 1.2x wider SL/TP
        atr_buffer_applied = False
        if self.atr_buffer_percent and self.atr_buffer_percent > 0:
            buffer_multiplier = 1 + (self.atr_buffer_percent / 100.0)
            
            # Widen both SL and TP by the buffer (maintains R:R ratio)
            original_sl_move = abs(price - stop_loss)
            original_tp_move = abs(take_profit - price)
            
            buffered_sl_move = original_sl_move * buffer_multiplier
            buffered_tp_move = original_tp_move * buffer_multiplier
            
            if position_type.name == 'LONG':
                stop_loss = price - buffered_sl_move
                take_profit = price + buffered_tp_move
            else:  # SHORT
                stop_loss = price + buffered_sl_move
                take_profit = price - buffered_tp_move
            
            atr_buffer_applied = True
            print(f"📊 ATR Buffer Applied: +{self.atr_buffer_percent}% (multiplier: {buffer_multiplier:.2f}x)")
        
        # MCX commodity lot sizes (common ones) - lookup table for mock mode
        mcx_lot_sizes = {
            'NATGASMINI': 250,  # MMBTU
            'GOLDM': 100,  # grams
            'SILVERM': 30,  # kg
            'CRUDEOILM': 100,  # barrels
            'SILVERMIC': 5,  # kg
            'GOLDGUINEA': 8,  # grams
        }
        
        # Get and store lot_size for commodity futures (needed for correct PnL calculation)
        # (is_kite_broker and is_mcx_commodity are already defined above)
        lot_size = None
        if is_kite_broker:
            print(f"🔍 Debug: Attempting to retrieve lot_size for {self.symbol}")
            try:
                # Try get_symbol_filters first
                print(f"🔍 Debug: Calling broker.get_symbol_filters('{self.symbol}')")
                lot_info = self.broker.get_symbol_filters(self.symbol)
                print(f"🔍 Debug: get_symbol_filters returned: {lot_info}")
                if lot_info:
                    lot_size = lot_info.get('lot_size')
                    if lot_size:
                        print(f"📦 Lot Size retrieved from get_symbol_filters: {lot_size} (for {self.symbol})")
                    else:
                        print(f"⚠️  lot_info exists but lot_size key not found. Keys: {list(lot_info.keys())}")
                
                # If not found, try get_lot_size method directly
                if not lot_size and hasattr(self.broker, 'get_lot_size'):
                    print(f"🔍 Debug: Trying broker.get_lot_size('{self.symbol}')")
                    lot_size = self.broker.get_lot_size(self.symbol)
                    print(f"🔍 Debug: get_lot_size returned: {lot_size}")
                    if lot_size:
                        print(f"📦 Lot Size retrieved from get_lot_size: {lot_size} (for {self.symbol})")
                
                if not lot_size:
                    print(f"⚠️  Could not retrieve lot_size for {self.symbol}")
            except Exception as e:
                print(f"⚠️  Error retrieving lot_size: {e}")
                import traceback
                traceback.print_exc()
        
        # If lot_size not found from broker, try lookup table for MCX commodities
        if not lot_size and is_mcx_commodity and hasattr(self, 'symbol') and self.symbol:
            for pattern, size in mcx_lot_sizes.items():
                if pattern in self.symbol.upper():
                    lot_size = size
                    print(f"📦 Lot Size from lookup table: {lot_size} (for {self.symbol}, pattern: {pattern})")
                    break
        
        if not lot_size and (is_kite_broker or is_mcx_commodity):
            print(f"⚠️  Could not determine lot_size for {self.symbol if hasattr(self, 'symbol') else 'symbol'}")
        
        # Only proceed with SL/TP validation if we have both lot_size and actual_lot_margin
        if is_kite_broker and actual_lot_margin and actual_lot_margin > 0 and lot_size:
            try:
                # Calculate potential loss with current stop loss (after buffer)
                sl_price_move = abs(price - stop_loss)
                sl_loss_rupees = lot_size * sl_price_move
                sl_loss_percent = (sl_loss_rupees / actual_lot_margin) * 100
                
                # Calculate max allowed loss
                max_loss_rupees = actual_lot_margin * (self.max_loss_percent / 100.0)
                max_price_move = max_loss_rupees / lot_size if lot_size > 0 else sl_price_move
                
                # Cap the stop loss if it exceeds max_loss_percent (even after buffer)
                sl_capped = False
                if sl_loss_rupees > max_loss_rupees:
                    sl_capped = True
                    if position_type.name == 'LONG':
                        stop_loss = price - max_price_move
                    else:  # SHORT
                        stop_loss = price + max_price_move
                
                # Log SL/TP levels
                actual_price_move = abs(price - stop_loss)
                actual_loss_rupees = lot_size * actual_price_move
                actual_loss_percent = (actual_loss_rupees / actual_lot_margin) * 100
                
                tp_price_move = abs(take_profit - price)
                tp_profit_rupees = lot_size * tp_price_move
                tp_profit_percent = (tp_profit_rupees / actual_lot_margin) * 100
                
                print(f"📊 ATR-based SL/TP (Lot size: {lot_size}, ATR: ₹{current_atr:.2f})")
                if atr_buffer_applied:
                    print(f"   📈 ATR Buffer: +{self.atr_buffer_percent}% wider SL/TP")
                if sl_capped:
                    print(f"   ⚠️ Stop Loss CAPPED: loss ({sl_loss_percent:.1f}%) exceeded max ({self.max_loss_percent}%)")
                print(f"   Stop Loss: ₹{stop_loss:.2f} (loss: ₹{actual_loss_rupees:.2f} = {actual_loss_percent:.1f}% of margin)")
                print(f"   Take Profit: ₹{take_profit:.2f} (profit: ₹{tp_profit_rupees:.2f} = {tp_profit_percent:.1f}% of margin)")
                
            except Exception as e:
                print(f"⚠️  Could not validate SL/TP: {e}. Using ATR-based levels.")
        
        # Try external broker execution if configured
        broker_order = None
        stop_loss_order = None
        gtt_order = None
        if self.broker and self.use_broker:
            try:
                # For Kite broker, use lot-based quantity (1 lot)
                # For other brokers, use calculated quantity
                order_quantity = 1 if is_kite_broker else quantity
                
                # Place MARKET order for immediate execution
                broker_order = self.broker.place_order(
                    symbol=self.symbol,
                    side='BUY' if position_type.name == 'LONG' else 'SELL',
                    order_type='MARKET',
                    quantity=order_quantity
                )
                order_id = broker_order.get('orderId') or broker_order.get('order_id')
                print(f"[Broker] Entry order placed: {order_id}")
                
                # For Kite broker, use GTT for stop-loss (persists overnight)
                # For other brokers, use regular stop-loss order
                if is_kite_broker:
                    # Determine transaction type for GTT
                    gtt_transaction_type = 'SELL' if position_type.name == 'LONG' else 'BUY'
                    
                    # Place GTT stop-loss order
                    gtt_order = self.broker.place_gtt_order(
                        symbol=self.symbol,
                        trigger_price=stop_loss,
                        last_price=price,
                        transaction_type=gtt_transaction_type,
                        quantity=order_quantity,
                        order_price=stop_loss
                    )
                    gtt_id = gtt_order.get('gtt_id') or gtt_order.get('trigger_id')
                    print(f"[Broker] GTT stop-loss order placed: {gtt_id} at ${stop_loss:.2f}")
                    stop_loss_order = gtt_order  # Store GTT order as stop_loss_order for compatibility
                else:
                    # Place regular stop-loss order for non-Kite brokers
                    stop_side = 'SELL' if position_type.name == 'LONG' else 'BUY'
                    stop_loss_order = self.broker.place_order(
                        symbol=self.symbol,
                        side=stop_side,
                        order_type='STOP_LOSS',
                        quantity=order_quantity,
                        price=stop_loss
                    )
                    stop_order_id = stop_loss_order.get('orderId') or stop_loss_order.get('order_id')
                    print(f"[Broker] Stop-loss order placed: {stop_order_id} at ${stop_loss:.2f}")
                
            except Exception as e:
                print(f"[Broker] Order placement failed, falling back to virtual fill: {e}")
                broker_order = None
                stop_loss_order = None
                gtt_order = None

        # Create trade record
        # Extract order IDs (handle both 'orderId' and 'order_id' formats)
        broker_order_id = None
        if isinstance(broker_order, dict):
            broker_order_id = broker_order.get('orderId') or broker_order.get('order_id')
        
        stop_loss_order_id = None
        gtt_id = None
        if isinstance(stop_loss_order, dict):
            # Check if it's a GTT order
            gtt_id = stop_loss_order.get('gtt_id') or stop_loss_order.get('trigger_id')
            stop_loss_order_id = stop_loss_order.get('orderId') or stop_loss_order.get('order_id') or gtt_id
        
        trade = {
            'id': len(self.trade_history) + 1,
            'strategy': strategy.name,
            'action': action, # BUY, SELL, EXIT
            'entry_price': price,
            'entry_time': timestamp,
            'quantity': quantity,
            'leverage': leverage,
            'position_size': position_size,
            'atr': current_atr,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'status': 'open',
            'pnl': 0,
            'broker_order_id': broker_order_id,
            'stop_loss_order_id': stop_loss_order_id,
            'gtt_id': gtt_id,  # Store GTT ID separately for Kite broker
            'lot_size': lot_size  # Store lot_size for commodity futures (needed for correct PnL calculation)
        }
        
        # Debug: Print lot_size storage
        if lot_size:
            print(f"💾 Stored lot_size in trade record: {lot_size}")
        else:
            print(f"⚠️  lot_size is None/not found - will use quantity only for PnL")
        
        # Update balance (subtract the margin requirement, not the full position size)
        # For Kite commodity trading, use actual lot margin from API
        # For other brokers, use calculated margin = position_size / leverage
        is_kite_broker = hasattr(self.broker, 'place_gtt_order') if self.broker else False
        
        if is_kite_broker and actual_lot_margin and actual_lot_margin > 0:
            # Use actual margin from Kite API for commodity trading
            margin_required = actual_lot_margin
            # Recalculate effective leverage based on actual margin
            effective_leverage = position_size / margin_required if margin_required > 0 else leverage
        else:
            # Standard margin calculation for other brokers
            margin_required = position_size / leverage
            effective_leverage = leverage
        
        self.current_balance -= margin_required
        
        # Store actual margin used in trade record for proper PnL calculation
        trade['margin_used'] = margin_required
        trade['effective_leverage'] = effective_leverage
        
        # Add to active trades
        self.active_trades.append(trade)
        self.trade_history.append(trade)
        
        # Log trade
        self._log_trade(trade, price, timestamp)
        
        # Print trade execution log
        print(f"🔄 [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {action} {quantity:.4f} {self.symbol} @ ${price:.2f}")
        print(f"💰 Position Size: ${position_size:.2f} (Leverage: {effective_leverage:.1f}x), Margin: ${margin_required:.2f}, New Balance: ${self.current_balance:.2f}")
        # Format ATR with appropriate precision: use 5 decimal places for small values (< 0.01), otherwise 2 decimal places
        if current_atr < 0.01:
            print(f"📊 ATR: ${current_atr:.5f}, Max Loss: {self.max_loss_percent}%")
        else:
            print(f"📊 ATR: ${current_atr:.2f}, Max Loss: {self.max_loss_percent}%")
        
        # Always print stop loss and take profit levels
        print(f"🛑 Stop Loss: ${stop_loss:.2f}")
        print(f"🎯 Take Profit: ${take_profit:.2f}")
        
        return trade
    
    def close_trades(self, strategy: BaseStrategy, position_type: str, price: float, timestamp: datetime, exit_type: str = None) -> List[Dict]:
        """
        Close existing trades.
        
        Args:
            strategy: Strategy instance
            position_type: Type of position to close ('LONG' or 'SHORT')
            price: Current price
            timestamp: Close timestamp
            exit_type: Optional exit type ('tp_hit', 'sl_hit', 'closed'). If None, will try to detect from strategy.
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.active_trades[:]:  # Copy list to avoid modification during iteration
            if trade['strategy'] == strategy.name and trade['status'] == 'open':
                if (position_type == 'LONG' and trade['action'] == 'BUY') or \
                   (position_type == 'SHORT' and trade['action'] == 'SELL'):
                    
                    # Calculate PnL with leverage
                    leverage = trade.get('effective_leverage', trade.get('leverage', 1.0))
                    position_size = trade.get('position_size', trade['quantity'] * trade['entry_price'])
                    
                    # For commodity futures (Kite broker), we use lot_size for PnL calculation.
                    # For spot/FX/CFD-style trading, we use quantity directly (no concept of exchange lot_size).
                    # lot_size represents the contract size per lot (e.g., 250 MMBTU for NATGASMINI)
                    # quantity represents number of units (FX/CFD) or number of lots (commodities).
                    lot_size_for_pnl = trade.get('lot_size')
                    quantity_for_pnl = trade.get('quantity', 1)
                    
                    print(f"🔍 PnL Calculation Debug:")
                    print(f"   quantity from trade: {quantity_for_pnl}")
                    print(f"   entry_price: {trade['entry_price']:.2f}")
                    print(f"   exit_price: {price:.2f}")
                    
                    if lot_size_for_pnl and lot_size_for_pnl > 0:
                        # Commodity futures: PnL = price_change × lot_size × quantity
                        # Since quantity is typically 1 for commodities, this simplifies to price_change × lot_size
                        print(f"   lot_size from trade (commodity contract size): {lot_size_for_pnl}")
                        if trade['action'] == 'BUY':
                            # For LONG positions: profit = (current_price - entry_price) * lot_size * quantity
                            price_change = price - trade['entry_price']
                            dollar_pnl = price_change * lot_size_for_pnl * quantity_for_pnl
                            print(f"   ✅ Commodity PnL: ({price:.2f} - {trade['entry_price']:.2f}) × {lot_size_for_pnl} × {quantity_for_pnl} = {dollar_pnl:.2f}")
                        else:
                            # For SHORT positions: profit = (entry_price - current_price) * lot_size * quantity
                            price_change = trade['entry_price'] - price
                            dollar_pnl = price_change * lot_size_for_pnl * quantity_for_pnl
                            print(f"   ✅ Commodity PnL: ({trade['entry_price']:.2f} - {price:.2f}) × {lot_size_for_pnl} × {quantity_for_pnl} = {dollar_pnl:.2f}")
                    else:
                        # FX/spot/CFD-style: quantity is already in base units, so we ignore lot_size completely.
                        if trade['action'] == 'BUY':
                            # LONG FX: profit = (exit_price - entry_price) * quantity
                            price_change = price - trade['entry_price']
                            dollar_pnl = price_change * quantity_for_pnl
                            print(f"   ✅ FX/Spot PnL: ({price:.2f} - {trade['entry_price']:.2f}) × {quantity_for_pnl} = {dollar_pnl:.2f}")
                        else:
                            # SHORT FX: profit = (entry_price - exit_price) * quantity
                            price_change = trade['entry_price'] - price
                            dollar_pnl = price_change * quantity_for_pnl
                            print(f"   ✅ FX/Spot PnL: ({trade['entry_price']:.2f} - {price:.2f}) × {quantity_for_pnl} = {dollar_pnl:.2f}")
                    
                    # Calculate percentage PnL based on ACTUAL margin used
                    # Use stored margin_used (from Kite API) if available, otherwise calculate
                    margin_used = trade.get('margin_used', position_size / leverage)
                    pnl = dollar_pnl / margin_used if margin_used > 0 else 0
                    
                    # Handle broker position closure
                    broker_exit_order = None
                    if self.broker and self.use_broker and trade.get('broker_order_id'):
                        # Determine exit status - use provided exit_type or try to detect from strategy
                        exit_status = exit_type
                        if exit_status is None:
                            strategy_trades = strategy.get_trade_history()
                            if not strategy_trades.empty:
                                recent_trades = strategy_trades.tail(1)
                                if not recent_trades.empty:
                                    exit_status = recent_trades['Status'].iloc[0]
                        
                        # Check if this is a Kite broker (has GTT support)
                        is_kite_broker = hasattr(self.broker, 'delete_gtt')
                        
                        # ALWAYS cancel/delete the GTT stop-loss order first (if exists)
                        # This prevents orphaned GTTs that could create unwanted positions
                        if is_kite_broker and trade.get('gtt_id'):
                            try:
                                self.broker.delete_gtt(trade['gtt_id'])
                                print(f"[Broker] GTT stop-loss order deleted: {trade['gtt_id']}")
                            except Exception as delete_e:
                                error_msg = str(delete_e).lower()
                                if "not found" in error_msg or "already" in error_msg or "triggered" in error_msg:
                                    print(f"[Broker] GTT {trade['gtt_id']} already deleted or triggered")
                                else:
                                    print(f"[Broker] Failed to delete GTT: {delete_e}")
                        elif trade.get('stop_loss_order_id') and not is_kite_broker:
                            # Cancel regular stop-loss order for non-Kite brokers
                            try:
                                open_orders = self.broker.get_open_orders(self.symbol)
                                order_exists = any(str(order.get('orderId')) == str(trade['stop_loss_order_id']) 
                                                 for order in open_orders)
                                if order_exists:
                                    self.broker.cancel_order(self.symbol, trade['stop_loss_order_id'])
                                    print(f"[Broker] Stop-loss order cancelled: {trade['stop_loss_order_id']}")
                            except Exception as cancel_e:
                                error_msg = str(cancel_e)
                                if "Unknown order sent" in error_msg or "-2011" in error_msg:
                                    print(f"[Broker] Stop-loss order {trade['stop_loss_order_id']} already filled or cancelled")
                                else:
                                    print(f"[Broker] Failed to cancel stop-loss order: {cancel_e}")
                        
                        # Now handle the exit order based on exit type
                        if exit_status == "sl_hit":
                            # Stop-loss was hit - broker already closed position (GTT triggered)
                            print(f"[Broker] Stop-loss hit detected - position already closed by exchange")
                        else:
                            # Take-profit hit, manual exit, or any other exit - place market order
                            try:
                                exit_side = 'SELL' if trade['action'] == 'BUY' else 'BUY'
                                # For Kite broker, use 1 lot; for others, use trade quantity
                                exit_quantity = 1 if is_kite_broker else trade['quantity']
                                broker_exit_order = self.broker.place_order(
                                    symbol=self.symbol,
                                    side=exit_side,
                                    order_type='MARKET',
                                    quantity=exit_quantity
                                )
                                exit_order_id = broker_exit_order.get('orderId') or broker_exit_order.get('order_id')
                                print(f"[Broker] Exit order placed: {exit_order_id} (exit_type: {exit_status or 'manual'})")
                            except Exception as e:
                                print(f"[Broker] Exit order failed, using virtual close: {e}")
                                broker_exit_order = None

                    # Update trade
                    trade['exit_price'] = price
                    trade['exit_time'] = timestamp
                    trade['pnl'] = pnl
                    trade['broker_exit_order_id'] = broker_exit_order.get('orderId') if isinstance(broker_exit_order, dict) else None
                    
                    # Get detailed status from strategy's trade history
                    strategy_trades = strategy.get_trade_history()
                    if not strategy_trades.empty:
                        # Find the most recent closed trade from the strategy
                        recent_trades = strategy_trades.tail(1)
                        if not recent_trades.empty:
                            detailed_status = recent_trades['Status'].iloc[0]
                            trade['status'] = detailed_status
                        else:
                            trade['status'] = 'closed'
                    else:
                        trade['status'] = 'closed'
                    
                    # Update balance - add back margin plus profit/loss
                    # For leveraged trades, we only used margin, not the full position value
                    # Use margin_used already calculated above (line 571) - this uses stored value from Kite API
                    # for commodity futures, ensuring correct margin calculation based on base price
                    
                    # Add back the margin plus the dollar PnL
                    self.current_balance += margin_used + dollar_pnl
                    
                    # Remove from active trades
                    self.active_trades.remove(trade)
                    closed_trades.append(trade)
                    
                    # Log trade closure
                    self._log_trade(trade, price, timestamp, is_exit=True)
                    
                    # Print trade closure log
                    print(f"✅ [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - CLOSED {trade['action']} position")
                    print(f"📈 PnL: {pnl:.2%} (${dollar_pnl:.2f})")
                    print(f"💰 Position Size: ${position_size:.2f}, Margin Used: ${margin_used:.2f}, Leverage: {leverage:.1f}x")
                    print(f"💰 New Balance: ${self.current_balance:.2f}")
        
        return closed_trades
    
    def process_strategy_signals(self, strategy: BaseStrategy, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Process strategy signals and execute trades.
        
        Args:
            strategy: Strategy instance
            data: Market data
            current_time: Current timestamp
            
        Returns:
            Dictionary containing signal processing results
        """
        # Generate signals
        signals_data = strategy.generate_signals(data)
        
        if signals_data is None or signals_data.empty:
            return {'signal': None, 'action': None, 'trades_executed': []}
            
        # Get the latest signal from the 'Signal' column
        latest_signal = int(signals_data['Signal'].iloc[-1])
        current_price = float(data['Close'].iloc[-1])
        
        # Log signal generation
        signal_name = {
            Signal.HOLD.value: "HOLD",
            Signal.LONG_ENTRY.value: "LONG_ENTRY", 
            Signal.SHORT_ENTRY.value: "SHORT_ENTRY",
            Signal.LONG_EXIT.value: "LONG_EXIT",
            Signal.SHORT_EXIT.value: "SHORT_EXIT"
        }.get(latest_signal, f"UNKNOWN({latest_signal})")
        
        #print(f"📊 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - Signal: {signal_name} @ ${current_price:.2f}")
        
        # Log decision to CSV
        self._log_decision(strategy, latest_signal, signal_name, current_price, current_time, signals_data)
        
        trades_executed = []
        action_taken = None
        
        # Check for entry signals (only if no active trades exist and sufficient balance)
        if latest_signal == Signal.LONG_ENTRY.value and self.current_balance > 0 and len(self.active_trades) == 0:  # Long entry
            print(f"🟢 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG ENTRY signal detected")
            trade = self.execute_trade(strategy, 'BUY', current_price, current_time, data)
            trades_executed.append(trade)
            action_taken = 'LONG_ENTRY'
        elif latest_signal == Signal.SHORT_ENTRY.value and self.current_balance > 0 and len(self.active_trades) == 0:  # Short entry
            print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT ENTRY signal detected")
            trade = self.execute_trade(strategy, 'SELL', current_price, current_time, data)
            trades_executed.append(trade)
            action_taken = 'SHORT_ENTRY'
        elif (latest_signal == Signal.LONG_ENTRY.value or latest_signal == Signal.SHORT_ENTRY.value) and len(self.active_trades) > 0:
            # Signal ignored because active trade exists
            signal_type = "LONG ENTRY" if latest_signal == Signal.LONG_ENTRY.value else "SHORT ENTRY"
            print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {signal_type} signal IGNORED (active trade exists)")
        elif (latest_signal == Signal.LONG_ENTRY.value or latest_signal == Signal.SHORT_ENTRY.value) and self.current_balance <= 0:
            # Signal ignored because insufficient balance
            signal_type = "LONG ENTRY" if latest_signal == Signal.LONG_ENTRY.value else "SHORT ENTRY"
            print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {signal_type} signal IGNORED (insufficient balance: ${self.current_balance:.2f})")
        
        # Check for exit signals
        if latest_signal == Signal.LONG_EXIT.value:  # Long exit
            if len(self.active_trades) > 0:
                print(f"🟢 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG EXIT signal detected")
                closed_trades = self.close_trades(strategy, 'LONG', current_price, current_time)
                trades_executed.extend(closed_trades)
                action_taken = 'LONG_EXIT'
            else:
                print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG EXIT signal IGNORED (no active trades)")
        elif latest_signal == Signal.SHORT_EXIT.value:  # Short exit
            if len(self.active_trades) > 0:
                print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT EXIT signal detected")
                closed_trades = self.close_trades(strategy, 'SHORT', current_price, current_time)
                trades_executed.extend(closed_trades)
                action_taken = 'SHORT_EXIT'
            else:
                print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT EXIT signal IGNORED (no active trades)")
        
        return {
            'signal': latest_signal,
            'signal_name': signal_name,
            'action': action_taken,
            'trades_executed': trades_executed,
            'current_price': current_price
        }
    
    def get_current_status(self, current_data: pd.DataFrame = None) -> Dict:
        """
        Get current trading status.
        
        Args:
            current_data: Current market data for unrealized PnL calculation
            
        Returns:
            Dictionary containing current trading status
        """
        # Calculate unrealized PnL from active trades
        unrealized_pnl = 0
        active_trade_info = None
        
        for trade in self.active_trades:
            if trade['status'] == 'open':
                # Calculate current value of the position
                current_price = current_data['Close'].iloc[-1] if current_data is not None and not current_data.empty else trade['entry_price']
                current_value = trade['quantity'] * current_price
                original_value = trade['quantity'] * trade['entry_price']
                
                if trade['action'] == 'BUY':
                    unrealized_pnl += current_value - original_value
                else:  # SELL (short)
                    unrealized_pnl += original_value - current_value
                
                # Store active trade info for display
                active_trade_info = {
                    'strategy': trade['strategy'],
                    'action': trade['action'],
                    'entry_price': trade['entry_price'],
                    'quantity': trade['quantity'],
                    'entry_time': trade['entry_time'],
                    # ATR at entry (useful for sizing + explaining risk)
                    'atr': trade.get('atr', None)
                }
        
        total_pnl = self.current_balance - self.initial_balance
        total_value = self.current_balance + unrealized_pnl
        
        return {
            'current_balance': self.current_balance,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_value': total_value,
            'active_trades': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'active_trade_info': active_trade_info,
            'can_trade': len(self.active_trades) == 0 and self.current_balance > 0
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame containing trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics from trade history.
        
        Returns:
            Dictionary containing performance metrics
        """
        trade_history_df = self.get_trade_history_df()
        
        if trade_history_df.empty:
            return self._get_empty_metrics()
        
        # Filter closed trades (including all closed statuses)
        closed_trades = trade_history_df[trade_history_df['status'].isin(['closed', 'tp_hit', 'sl_hit', 'reversed'])].copy()
        
        if closed_trades.empty:
            print(f"[INFO] No closed trades found in trade history")
            return self._get_empty_metrics()
        
        # Sort by exit time for proper chronological order
        closed_trades = closed_trades.sort_values('exit_time')
        
        # Separate winning and losing trades
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        # Basic metrics
        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_return = closed_trades['pnl'].mean() if total_trades > 0 else 0
        
        # Calculate total PnL based on actual balance change
        # This matches the actual balance progression in the system
        total_pnl = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Geometric mean return (average return per trade)
        # Calculate using compounding of individual trade returns
        total_multiplier = 1.0
        for _, trade in closed_trades.iterrows():
            total_multiplier *= (1 + trade['pnl'])
        geometric_mean_return = (total_multiplier ** (1.0 / total_trades)) - 1.0 if total_trades > 0 else 0
        
        # Win/Loss metrics
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        largest_win = closed_trades['pnl'].max() if total_trades > 0 else 0
        largest_loss = closed_trades['pnl'].min() if total_trades > 0 else 0
        
        # Risk-reward ratio (win/loss ratio)
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        # Profit factor
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf') if total_profit > 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for _, trade in closed_trades.iterrows():
            if trade['pnl'] > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        # Sharpe ratio
        returns = closed_trades['pnl'].tolist()
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe_ratio = mean_return / max(std_return, 1e-8)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns, ddof=1)
                sortino_ratio = mean_return / max(downside_deviation, 1e-8)
            else:
                sortino_ratio = float('inf') if mean_return > 0 else 0
        else:
            sortino_ratio = 0
        
        # Max drawdown
        cumulative_returns = []
        cumulative_return = 1.0
        for _, trade in closed_trades.iterrows():
            cumulative_return *= (1 + trade['pnl'])
            cumulative_returns.append(cumulative_return)
        
        max_drawdown = 0
        peak = 1.0
        for cr in cumulative_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar ratio
        if len(closed_trades) > 1:
            first_trade_date = closed_trades['exit_time'].min()
            last_trade_date = closed_trades['exit_time'].max()
            trading_days = (last_trade_date - first_trade_date).days
            if trading_days > 0:
                annualized_return = (total_multiplier ** (365 / trading_days)) - 1
            else:
                annualized_return = (1 + geometric_mean_return) ** 252 - 1
        else:
            annualized_return = (1 + geometric_mean_return) ** 252 - 1
        
        calmar_ratio = annualized_return / max(max_drawdown, 1e-8)
        
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
            "risk_reward_ratio": risk_reward_ratio,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio
        }
    
    def _get_empty_metrics(self) -> Dict:
        """
        Get empty metrics structure.
        
        Returns:
            Dictionary with zero values for all metrics
        """
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
    
    def _log_trade(self, trade: Dict, price: float, timestamp: datetime, is_exit: bool = False):
        """
        Log trade to CSV file.
        
        Args:
            trade: Trade dictionary
            price: Trade price
            timestamp: Trade timestamp
            is_exit: Whether this is an exit trade
        """
        if self.trade_log_file:
            # Convert timestamp to IST timezone to match decision logs
            ist = pytz.timezone('Asia/Kolkata')
            if timestamp.tzinfo is None:
                # If no timezone info, assume UTC and convert to IST
                timestamp_ist = pytz.utc.localize(timestamp).astimezone(ist)
            else:
                # If already has timezone, convert to IST
                timestamp_ist = timestamp.astimezone(ist)
            
            log_entry = {
                'timestamp': timestamp_ist,
                'symbol': self.symbol,
                'strategy': trade['strategy'],
                'action': 'EXIT' if is_exit else trade['action'],
                'price': price,
                'quantity': trade['quantity'],
                'leverage': trade.get('leverage', 1.0),
                'position_size': trade.get('position_size', trade['quantity'] * price),
                'atr': trade.get('atr', 0.0),
                'balance': self.current_balance,
                'pnl': trade.get('pnl', 0),
                'trade_id': trade['id'],
                'status': trade['status'],
                'reject_reason': trade.get('reject_reason', '')
            }
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
    
    def _log_decision(self, strategy: BaseStrategy, signal: int, signal_name: str, 
                     current_price: float, current_time: datetime, signals_data: pd.DataFrame):
        """
        Log algorithm decision to CSV file.
        
        Args:
            strategy: Strategy instance
            signal: Signal value
            signal_name: Human-readable signal name
            current_price: Current price
            current_time: Current timestamp
            signals_data: Signals data from strategy
        """
        if self.decision_log_file:
            # Get current position info
            active_trades_count = len(self.active_trades)
            position_type = "NONE"
            take_profit = None
            stop_loss = None
            
            # Get take profit and stop loss from signals data if available
            if 'Take_Profit' in signals_data.columns and 'Stop_Loss' in signals_data.columns:
                take_profit = signals_data['Take_Profit'].iloc[-1]
                stop_loss = signals_data['Stop_Loss'].iloc[-1]
                
                # Handle NaN values
                if pd.isna(take_profit):
                    take_profit = None
                if pd.isna(stop_loss):
                    stop_loss = None
            
            # Find active trade for this strategy
            trade_status = "NONE"
            for trade in self.active_trades:
                if trade['strategy'] == strategy.name:
                    position_type = trade['action']
                    trade_status = trade.get('status', 'open')
                    break
            
            # Convert timestamp to IST timezone to match market data
            ist = pytz.timezone('Asia/Kolkata')
            if current_time.tzinfo is None:
                # If no timezone info, assume UTC and convert to IST
                current_time_ist = pytz.utc.localize(current_time).astimezone(ist)
            else:
                # If already has timezone, convert to IST
                current_time_ist = current_time.astimezone(ist)
            
            log_entry = {
                'timestamp': current_time_ist,
                'symbol': self.symbol,
                'strategy': strategy.name,
                'signal': signal,
                'signal_name': signal_name, # HOLD, LONG_ENTRY, SHORT_ENTRY, LONG_EXIT, SHORT_EXIT
                'current_price': current_price,
                'current_balance': self.current_balance,
                'active_trades_count': active_trades_count,
                'position_type': position_type,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'trade_status': trade_status
            }
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.decision_log_file, mode='a', header=False, index=False)
    
    def save_data_to_csv(self, data: pd.DataFrame, fetch_time: datetime):
        """
        Save fetched data to CSV file.
        
        Args:
            data: Market data
            fetch_time: Data fetch timestamp
        """
        if self.data_log_file and not data.empty:
            # Add fetch timestamp to data
            data_copy = data.copy()
            data_copy['fetch_timestamp'] = fetch_time
            
            # Save to CSV (append mode)
            data_copy.to_csv(self.data_log_file, mode='a', header=not os.path.exists(self.data_log_file), index=True)
            
            # print(f"💾 [{fetch_time.strftime('%Y-%m-%d %H:%M:%S')}] Data saved to {self.data_log_file}")
    
    def get_log_files_info(self) -> Dict:
        """
        Get information about log files.
        
        Returns:
            Dictionary containing log file paths
        """
        return {
            'trade_log': self.trade_log_file,
            'data_log': self.data_log_file,
            'decision_log': self.decision_log_file,
            'session_id': self.session_id
        }
    
    def get_decision_log_df(self) -> pd.DataFrame:
        """
        Get decision log as DataFrame.
        
        Returns:
            DataFrame containing decision log
        """
        if not self.decision_log_file or not os.path.exists(self.decision_log_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.decision_log_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error reading decision log: {e}")
            return pd.DataFrame()
    
    def reset(self):
        """
        Reset the trading engine to initial state.
        """
        self.current_balance = self.initial_balance
        self.trade_history = []
        self.active_trades = []
        self.total_pnl = 0
        self.unrealized_pnl = 0
        print("🔄 Trading engine reset to initial state.")
