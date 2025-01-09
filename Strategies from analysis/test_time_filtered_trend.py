import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from time_filtered_trend import TimeFilteredTrendStrategy
import mplfinance as mpf

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Ensure data is sorted by datetime
    df = df.sort_values(by='datetime')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    return df

def test_time_filtered_trend_strategy(file_path: str):
    """Test Time Filtered Trend Strategy indicators calculation and visualization"""
    # Load market data
    print(f"Loading market data from {file_path}...")
    df = load_market_data(file_path)
    df.set_index('datetime', inplace=True)
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create Backtrader cerebro instance
    cerebro = bt.Cerebro()
    
    # Add data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Already indexed
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(TimeFilteredTrendStrategy)
    
    # Run strategy
    print("Running strategy...")
    results = cerebro.run()
    strategy = results[0]
    
    # Create plot data
    df_plot = df.copy()
    apds = []
    
    # Plot trade zones
    if strategy.trade_exits:
        for trade in strategy.trade_exits:
            try:
                # Get entry/exit data
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                
                is_long = 'long' in trade['type']
                is_successful = (is_long and exit_price > entry_price) or (not is_long and exit_price < entry_price)
                
                # Create connecting line between entry and exit
                trade_line = pd.Series(index=df_plot.index, data=np.nan)
                mask = (df_plot.index >= entry_time) & (df_plot.index <= exit_time)
                idx = df_plot.index[mask]
                
                if len(idx) >= 2:
                    trade_line.loc[entry_time] = entry_price
                    trade_line.loc[exit_time] = exit_price
                    trade_line.loc[entry_time:exit_time] = np.linspace(
                        entry_price, 
                        exit_price, 
                        len(df_plot.loc[entry_time:exit_time])
                    )
                
                # Add the connecting line
                color = 'green' if is_successful else 'red'
                apds.append(mpf.make_addplot(
                    trade_line,
                    type='line',
                    color=color,
                    width=1.5,
                    alpha=0.8
                ))

                # Plot entry point
                entry_series = pd.Series(index=df_plot.index, data=np.nan)
                entry_series[entry_time] = entry_price
                apds.append(
                    mpf.make_addplot(
                        entry_series,
                        type='scatter',
                        marker='^' if is_long else 'v',
                        markersize=100,
                        color='green' if is_long else 'red'
                    )
                )

                # Plot exit point
                exit_series = pd.Series(index=df_plot.index, data=np.nan)
                exit_series[exit_time] = exit_price
                apds.append(
                    mpf.make_addplot(
                        exit_series,
                        type='scatter',
                        marker='x',
                        markersize=100,
                        color='green' if is_successful else 'red'
                    )
                )

            except Exception as e:
                print(f"Warning: Could not plot trade: {e}")
                print(f"Trade data: {trade}")
                continue

    # Create custom style
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(up='green', down='red', inherit=True),
        gridcolor='gray',
        gridstyle='--',
        gridaxis='both'
    )

    # Plot with returnfig=True to get figure and axes
    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        title='Time Filtered Trend Strategy Trades',
        addplot=apds,
        volume=True,
        figsize=(20, 10),
        warn_too_much_data=100000,
        returnfig=True
    )

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                  markersize=10, label='Long Entry'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
                  markersize=10, label='Short Entry'),
        plt.Line2D([0], [0], marker='x', color='green', markersize=10, 
                  label='Profit Exit'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=10, 
                  label='Loss Exit'),
        plt.Line2D([0], [0], color='green', linewidth=3, label='Winning Trade'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Losing Trade')
    ]
    
    axes[0].legend(handles=legend_elements, loc='upper left')
    plt.show()

    # Print strategy statistics
    print("\nStrategy Statistics:")
    print(f"Total Trades: {len(strategy.trade_exits)}")
    
    if strategy.trade_exits:
        winning_trades = sum(1 for trade in strategy.trade_exits if trade['pnl'] > 0)
        total_pnl = sum(trade['pnl'] for trade in strategy.trade_exits)
        win_rate = (winning_trades / len(strategy.trade_exits)) * 100
        
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_time_filtered_trend_strategy(file_path) 