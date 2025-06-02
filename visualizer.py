import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict
from strategies import BaseStrategy, MovingAverageCrossover, RSIStrategy

class Visualizer:
    def __init__(self):
        self.fig = None
    
    def create_chart(self, data: pd.DataFrame, strategies: List[BaseStrategy]) -> go.Figure:
        """
        Create an interactive chart with OHLC data and strategy signals
        
        Args:
            data (pd.DataFrame): OHLCV data with signals
            strategies (List[BaseStrategy]): List of strategy objects
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create figure with 3 subplots
        self.fig = make_subplots(rows=3, cols=1, 
                                shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.5, 0.25, 0.25])
        
        # Add candlestick chart
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                hoverinfo='all',
                hovertext=[
                    f"Date: {date}<br>"
                    f"Open: ${open:.2f}<br>"
                    f"High: ${high:.2f}<br>"
                    f"Low: ${low:.2f}<br>"
                    f"Close: ${close:.2f}<br>"
                    f"Return: {ret:.2%}"
                    for date, open, high, low, close, ret in zip(
                        data.index,
                        data['Open'],
                        data['High'],
                        data['Low'],
                        data['Close'],
                        data['Returns']
                    )
                ]
            ),
            row=1, col=1
        )
        
        # Add volume bars
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                hovertemplate=(
                    "<b>Volume</b><br>" +
                    "Date: %{x}<br>" +
                    "Volume: %{y:,.0f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=1
        )
        
        # Add key points markers and annotations
        key_points = data[data['Key_Point'] == True]
        
        # Add markers for key points
        self.fig.add_trace(
            go.Scatter(
                x=key_points.index,
                y=key_points['Close'],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='yellow',
                    line=dict(color='black', width=1)
                ),
                name='Key Point',
                hovertemplate=(
                    "<b>Key Point</b><br>" +
                    "Date: %{x}<br>" +
                    "Open: $%{customdata[0]:.2f}<br>" +
                    "High: $%{customdata[1]:.2f}<br>" +
                    "Low: $%{customdata[2]:.2f}<br>" +
                    "Close: $%{customdata[3]:.2f}<br>" +
                    "Return: %{customdata[4]:.2%}<br>" +
                    "<extra></extra>"
                ),
                customdata=key_points[['Open', 'High', 'Low', 'Close', 'Returns']].values
            ),
            row=1, col=1
        )
        
        # Add strategy signals and indicators
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        
        for i, strategy in enumerate(strategies):
            strategy_data = strategy.generate_signals(data)
            
            # Plot buy signals
            buy_signals = strategy_data[strategy_data['Signal'] == 1]
            self.fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=colors[i % len(colors)]
                    ),
                    name=f'{strategy.name} Buy',
                    hovertemplate=(
                        f"<b>{strategy.name} Buy Signal</b><br>" +
                        "Date: %{x}<br>" +
                        "Open: $%{customdata[0]:.2f}<br>" +
                        "High: $%{customdata[1]:.2f}<br>" +
                        "Low: $%{customdata[2]:.2f}<br>" +
                        "Close: $%{customdata[3]:.2f}<br>" +
                        "Return: %{customdata[4]:.2%}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=buy_signals[['Open', 'High', 'Low', 'Close', 'Returns']].values
                ),
                row=1, col=1
            )
            
            # Plot sell signals
            sell_signals = strategy_data[strategy_data['Signal'] == -1]
            self.fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=colors[i % len(colors)]
                    ),
                    name=f'{strategy.name} Sell',
                    hovertemplate=(
                        f"<b>{strategy.name} Sell Signal</b><br>" +
                        "Date: %{x}<br>" +
                        "Open: $%{customdata[0]:.2f}<br>" +
                        "High: $%{customdata[1]:.2f}<br>" +
                        "Low: $%{customdata[2]:.2f}<br>" +
                        "Close: $%{customdata[3]:.2f}<br>" +
                        "Return: %{customdata[4]:.2%}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=sell_signals[['Open', 'High', 'Low', 'Close', 'Returns']].values
                ),
                row=1, col=1
            )
            
            # Add strategy-specific indicators
            if isinstance(strategy, MovingAverageCrossover):
                # Plot Moving Averages on the OHLC chart
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=strategy_data['SMA_short'],
                        name='Short MA',
                        line=dict(color='blue', width=1),
                        hovertemplate=(
                            "<b>Short MA</b><br>" +
                            "Date: %{x}<br>" +
                            "Value: $%{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=1
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=strategy_data['SMA_long'],
                        name='Long MA',
                        line=dict(color='red', width=1),
                        hovertemplate=(
                            "<b>Long MA</b><br>" +
                            "Date: %{x}<br>" +
                            "Value: $%{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=1
                )
            
            elif isinstance(strategy, RSIStrategy):
                # Plot RSI
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=strategy_data['RSI'],
                        name='RSI',
                        line=dict(color='purple'),
                        hovertemplate=(
                            "<b>RSI</b><br>" +
                            "Date: %{x}<br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )
                
                # Add overbought/oversold lines
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=[strategy.overbought] * len(data),
                        name='Overbought',
                        line=dict(color='red', dash='dash'),
                        hovertemplate=(
                            "<b>Overbought Level</b><br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=[strategy.oversold] * len(data),
                        name='Oversold',
                        line=dict(color='green', dash='dash'),
                        hovertemplate=(
                            "<b>Oversold Level</b><br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )
        
        # Add vertical lines for key points
        for idx in key_points.index:
            self.fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                row=1, col=1
            )
            self.fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                row=2, col=1
            )
            self.fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                row=3, col=1
            )
        
        # Update layout
        self.fig.update_layout(
            title='Stock Price and Trading Signals',
            yaxis_title='Price',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            xaxis_rangeslider_visible=False,
            height=900,
            # Adjust margins to accommodate annotations
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update y-axis ranges for RSI
        self.fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        return self.fig
    
    def show(self):
        """Display the chart"""
        if self.fig is not None:
            self.fig.show() 