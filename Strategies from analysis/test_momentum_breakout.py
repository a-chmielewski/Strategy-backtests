import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
import backtrader as bt
from momentum_breakout import MomentumBreakoutStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_momentum_breakout_strategy(file_path: str):
    """Test Momentum Breakout Strategy with trade visualization"""
    # Load market data
    print(f"Loading market data from {file_path}...")
    df = load_market_data(file_path)
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
    cerebro.addstrategy(MomentumBreakoutStrategy)
    
    # Add broker settings
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        leverage=10,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    cerebro.broker.set_slippage_perc(0.0001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    
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
                if 'entry_time' in trade and 'exit_time' in trade:
                    entry_time = trade['entry_time']
                    exit_time = trade['exit_time']
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                else:
                    continue

                is_long = 'long' in trade['type']
                # Determine if trade was successful based on direction
                is_successful = (is_long and exit_price > entry_price) or (not is_long and exit_price < entry_price)
                
                # Create filled rectangle for the trade
                mask = (df_plot.index >= entry_time) & (df_plot.index <= exit_time)
                trade_period = df_plot.index[mask]
                
                if len(trade_period) == 0:
                    continue
                    
                # Create a series for the connecting line
                trade_line = pd.Series(index=df_plot.index, data=np.nan)
                
                # Create line points to connect entry and exit
                idx = df_plot.index[mask]
                if len(idx) >= 2:
                    # Only fill values between entry and exit times
                    trade_line.loc[entry_time] = entry_price
                    trade_line.loc[exit_time] = exit_price
                    # Fill intermediate points with linear interpolation
                    trade_line.loc[entry_time:exit_time] = np.linspace(
                        entry_price, 
                        exit_price, 
                        len(df_plot.loc[entry_time:exit_time])
                    )
                
                # Add the connecting line - color based on trade success
                color = 'green' if is_successful else 'red'
                apds.append(mpf.make_addplot(
                    trade_line,
                    type='line',
                    color=color,
                    width=1.5,
                    alpha=0.8
                ))

                # Plot entry point - color based on direction
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

                # Plot exit point - color based on trade success
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
        title='Momentum Breakout Strategy Trades',
        addplot=apds,
        volume=False,
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
    
    # Print summary statistics
    trades_analysis = strategy.analyzers.trades.get_analysis()
    
    total_trades = trades_analysis.total.closed if hasattr(trades_analysis, 'total') else 0
    winning_trades = trades_analysis.won.total if hasattr(trades_analysis, 'won') else 0
    total_pnl = trades_analysis.pnl.net.total if hasattr(trades_analysis, 'pnl') else 0
    
    print("\nStrategy Summary:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    if total_trades > 0:
        print(f"Win Rate: {(winning_trades/total_trades*100):.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    
    try:
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: N/A")
    except (KeyError, AttributeError):
        print("Sharpe Ratio: N/A")

    try:
        max_drawdown = strategy.analyzers.drawdown.get_analysis().max.drawdown
        print(f"Max Drawdown: {max_drawdown:.2f}%" if max_drawdown is not None else "Max Drawdown: N/A")
    except (KeyError, AttributeError):
        print("Max Drawdown: N/A")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_momentum_breakout_strategy(file_path) 