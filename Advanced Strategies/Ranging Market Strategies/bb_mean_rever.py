import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer
from results_logger import log_result

LEVERAGE = 50

class BB_MeanReversion_Strategy(bt.Strategy):
    params = (
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("volume_period", 20),
        ("min_range_width", 0.01),  # Minimum 1% range width to trade
        ("stop_loss_pct", 0.005),  # 0.5% stop outside range
        ("take_profit_mid", 0.8),  # Take 80% profit at middle band
        ("take_profit_far", 0.2),  # Keep 20% for opposite band
        ("time_stop_bars", 30),  # Max holding period for range trades
        ("trend_detection_period", 50),  # Period to detect trending vs ranging
        ("max_trend_slope", 0.002),  # Max slope to consider ranging (0.2%)
        ("min_volume_ratio", 0.8),  # Volume should be below average at extremes
        ("reversal_confirmation_bars", 2),  # Bars to confirm reversal
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.bb_period, 
            devfactor=self.p.bb_std
        )
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_period)
        
        # Bollinger Band components
        self.bb_top = self.bb.top
        self.bb_mid = self.bb.mid  # This is the 20-period SMA
        self.bb_bot = self.bb.bot
        
        # Trend detection for ranging vs trending markets
        self.trend_sma = bt.indicators.SMA(self.data.close, period=self.p.trend_detection_period)
        
        # Range state tracking
        self.in_range_mode = True
        self.range_high = 0
        self.range_low = float('inf')
        self.band_touch_detected = False
        self.touch_direction = 0  # 1 for upper band, -1 for lower band
        self.touch_bar = None
        self.waiting_for_reversal = False
        self.reversal_confirmed = False
        
        # Track recent price action for pattern recognition
        self.recent_highs = []
        self.recent_lows = []
        self.consolidation_bars = 0
        
        # Order and position tracking
        self.order = None
        self.order_parent_ref = None
        self.trade_dir = {}
        self.entry_bar = None
        self.entry_side = None
        self.closing = False
        self.partial_exit_done = False
        self.runner_position = 0

    def calculate_position_size(self, current_price):
        try:
            if current_price is None or current_price == 0:
                return 0
            current_equity = self.broker.getvalue()

            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0

            leverage = LEVERAGE

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def detect_market_regime(self):
        """Detect if market is ranging or trending"""
        if len(self) < self.p.trend_detection_period:
            return True  # Assume ranging until we have enough data
            
        # Calculate slope of long-term SMA
        current_sma = self.trend_sma[0]
        past_sma = self.trend_sma[-10] if len(self.trend_sma) > 10 else self.trend_sma[-1]
        
        slope = (current_sma - past_sma) / past_sma if past_sma != 0 else 0
        
        # Check if slope is within ranging threshold
        is_ranging = abs(slope) < self.p.max_trend_slope
        
        # Additional check: Bollinger Band width
        bb_width = (self.bb_top[0] - self.bb_bot[0]) / self.bb_mid[0] if self.bb_mid[0] != 0 else 0
        sufficient_width = bb_width > self.p.min_range_width
        
        return is_ranging and sufficient_width

    def detect_band_touch(self):
        """Detect when price touches Bollinger Bands with proper conditions"""
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # Volume confirmation (lower volume at extremes)
        volume_ok = True
        if hasattr(self.data, 'volume') and len(self.volume_sma) > 0:
            volume_ok = self.data.volume[0] < self.volume_sma[0] * self.p.min_volume_ratio
        
        # Check for upper band touch with overbought RSI
        if (current_high >= self.bb_top[0] or current_price > self.bb_top[0]) and self.rsi[0] > self.p.rsi_overbought:
            if volume_ok:
                self.band_touch_detected = True
                self.touch_direction = 1  # Upper band touch
                self.touch_bar = len(self)
                self.waiting_for_reversal = True
                return True
                
        # Check for lower band touch with oversold RSI
        elif (current_low <= self.bb_bot[0] or current_price < self.bb_bot[0]) and self.rsi[0] < self.p.rsi_oversold:
            if volume_ok:
                self.band_touch_detected = True
                self.touch_direction = -1  # Lower band touch
                self.touch_bar = len(self)
                self.waiting_for_reversal = True
                return True
        
        return False

    def check_reversal_confirmation(self):
        """Check for reversal confirmation after band touch"""
        if not self.waiting_for_reversal:
            return False, None
            
        current_price = self.data.close[0]
        bars_since_touch = len(self) - self.touch_bar if self.touch_bar else 0
        
        # Don't wait too long for reversal
        if bars_since_touch > self.p.reversal_confirmation_bars + 2:
            self.waiting_for_reversal = False
            self.band_touch_detected = False
            return False, None
        
        if self.touch_direction == 1:  # Upper band touch, looking for bearish reversal
            # Check if price closes back inside bands
            price_inside = current_price < self.bb_top[0]
            
            # Look for bearish reversal patterns
            prev_close = self.data.close[-1] if len(self) > 1 else current_price
            prev_open = self.data.open[-1] if len(self) > 1 else current_price
            current_open = self.data.open[0]
            
            # Bearish engulfing or shooting star patterns
            bearish_engulfing = (prev_close > prev_open and 
                               current_price < current_open and 
                               current_price < prev_open and 
                               current_open > prev_close)
            
            shooting_star = (current_open < current_price < self.data.high[0] and 
                           (self.data.high[0] - current_price) > 2 * (current_price - current_open))
            
            if price_inside and (bearish_engulfing or shooting_star or bars_since_touch >= self.p.reversal_confirmation_bars):
                return True, 'short'
                
        elif self.touch_direction == -1:  # Lower band touch, looking for bullish reversal
            # Check if price closes back inside bands
            price_inside = current_price > self.bb_bot[0]
            
            # Look for bullish reversal patterns
            prev_close = self.data.close[-1] if len(self) > 1 else current_price
            prev_open = self.data.open[-1] if len(self) > 1 else current_price
            current_open = self.data.open[0]
            
            # Bullish engulfing or hammer patterns
            bullish_engulfing = (prev_close < prev_open and 
                               current_price > current_open and 
                               current_price > prev_open and 
                               current_open < prev_close)
            
            hammer = (current_open > current_price > self.data.low[0] and 
                     (current_open - self.data.low[0]) > 2 * (current_open - current_price))
            
            if price_inside and (bullish_engulfing or hammer or bars_since_touch >= self.p.reversal_confirmation_bars):
                return True, 'long'
        
        return False, None

    def check_mean_reversion_exit(self):
        """Check for mean reversion exit conditions"""
        if not self.position:
            return False
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time stop
        if bars_held >= self.p.time_stop_bars:
            return True
            
        # Check for mean reversion to middle band
        if pos.size > 0:  # Long position
            # Take profit at middle band
            if current_price >= self.bb_mid[0]:
                return True
            # Stop loss if breaks below lower band again
            elif current_price < self.bb_bot[0]:
                return True
                
        elif pos.size < 0:  # Short position
            # Take profit at middle band
            if current_price <= self.bb_mid[0]:
                return True
            # Stop loss if breaks above upper band again
            elif current_price > self.bb_top[0]:
                return True
        
        return False

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.bb_period, self.p.rsi_period, self.p.trend_detection_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Step 1: Check if market is in ranging mode
        self.in_range_mode = self.detect_market_regime()
        
        if not self.in_range_mode:
            # Market is trending, avoid mean reversion trades
            if self.position:
                # Exit any existing position if market starts trending
                self.close()
                self.closing = True
                self.entry_bar = None
                self.entry_side = None
            return
        
        if not self.position:  # If no position is open
            # Step 2: Look for band touches
            if not self.band_touch_detected:
                self.detect_band_touch()
            
            # Step 3: Check for reversal confirmation
            reversal_signal, direction = self.check_reversal_confirmation()
            
            if reversal_signal and direction:
                position_size = self.calculate_position_size(current_price)
                
                if position_size <= 0:
                    return
                
                if direction == 'long':
                    # Entry near lower band, target middle band
                    stop_price = self.bb_bot[0] * (1 - self.p.stop_loss_pct)
                    target_price = self.bb_mid[0]
                    
                    # Split position for partial profit taking
                    main_size = position_size * self.p.take_profit_mid
                    runner_size = position_size * self.p.take_profit_far
                    
                    # Enter main position with target at middle band
                    parent, stop, limit = self.buy_bracket(
                        size=main_size,
                        exectype=bt.Order.Market,
                        stopprice=stop_price,
                        limitprice=target_price,
                    )
                    self.order = parent
                    self.order_parent_ref = parent.ref
                    self.entry_bar = len(self)
                    self.entry_side = 'long'
                    
                    # Store runner position size for manual management
                    self.runner_position = runner_size
                    
                elif direction == 'short':
                    # Entry near upper band, target middle band
                    stop_price = self.bb_top[0] * (1 + self.p.stop_loss_pct)
                    target_price = self.bb_mid[0]
                    
                    # Split position for partial profit taking
                    main_size = position_size * self.p.take_profit_mid
                    runner_size = position_size * self.p.take_profit_far
                    
                    # Enter main position with target at middle band
                    parent, stop, limit = self.sell_bracket(
                        size=main_size,
                        exectype=bt.Order.Market,
                        stopprice=stop_price,
                        limitprice=target_price,
                    )
                    self.order = parent
                    self.order_parent_ref = parent.ref
                    self.entry_bar = len(self)
                    self.entry_side = 'short'
                    
                    # Store runner position size for manual management
                    self.runner_position = runner_size
                
                # Reset reversal detection
                self.waiting_for_reversal = False
                self.band_touch_detected = False
                self.reversal_confirmed = False
        else:
            # Check for exit conditions
            should_exit = self.check_mean_reversion_exit()
            
            if should_exit:
                self.close()
                self.closing = True
                self.entry_bar = None
                self.entry_side = None
                self.partial_exit_done = False
                self.runner_position = 0

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            self.trade_dir[trade.ref] = 'long' if trade.size > 0 else 'short'
        if not trade.isclosed:
            return

        try:
            # Get entry and exit prices
            entry_price = trade.price
            exit_price = trade.history[-1].price if trade.history else self.data.close[0]
            pnl = trade.pnl
            
            # Store trade exit information for visualization
            direction = self.trade_dir.get(trade.ref, None)
            trade_type = f'{direction}_exit' if direction in ('long', 'short') else 'unknown_exit'
            self.trade_exits.append({
                'entry_time': trade.dtopen,
                'exit_time': trade.dtclose,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': trade_type,
                'pnl': pnl
            })
            
            if trade.ref in self.trade_dir:
                del self.trade_dir[trade.ref]

        except Exception as e:
            print(f"Warning: Could not process trade: {str(e)}")
            print(f"Trade info - Status: {trade.status}, Size: {trade.size}, "
                  f"Price: {trade.price}, PnL: {trade.pnl}")

    def notify_order(self, order):
        # Only clear self.order if this is the parent (entry) order and matches entry side
        if self.order and order.ref == self.order.ref and order.parent is None and (
            (self.entry_side == 'long' and order.isbuy()) or (self.entry_side == 'short' and order.issell())
        ) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
            if hasattr(self, 'order_parent_ref') and order.ref == self.order_parent_ref:
                self.order_parent_ref = None
        
        # Clear closing flag if a close order is done
        if getattr(self, 'closing', False) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.closing = False
            
        if order.status == order.Completed:
            if not order.parent:  # This is an entry order
                # Record trade start
                self.active_trades.append({
                    'entry_time': self.data.datetime.datetime(0),
                    'entry_price': order.executed.price,
                    'type': 'long' if order.isbuy() else 'short',
                    'size': order.executed.size
                })
            else:  # This is an exit order
                if self.active_trades:
                    trade = self.active_trades.pop()
                    # Record trade exit
                    self.trade_exits.append({
                        'entry_time': trade['entry_time'],
                        'entry_price': trade['entry_price'],
                        'exit_time': self.data.datetime.datetime(0),
                        'exit_price': order.executed.price,
                        'type': f"{trade['type']}_exit",
                        'pnl': (order.executed.price - trade['entry_price']) * trade['size'] if trade['type'] == 'long' 
                              else (trade['entry_price'] - order.executed.price) * trade['size']
                    })

def run_backtest(data, verbose=True, leverage=LEVERAGE, **kwargs):
    # Avoid zero-range bars: ensure High > Low on every bar
    mask = data["High"] <= data["Low"]
    if mask.any():
        data.loc[mask, "High"] = data.loc[mask, "Low"] + 1e-8
        
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(
        dataname=data,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        datetime="datetime",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=None,
    )
    cerebro.adddata(feed)
    
    strategy_params = {
        "bb_period": kwargs.get("bb_period", 20),
        "bb_std": kwargs.get("bb_std", 2.0),
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "volume_period": kwargs.get("volume_period", 20),
        "min_range_width": kwargs.get("min_range_width", 0.01),
        "stop_loss_pct": kwargs.get("stop_loss_pct", 0.005),
        "take_profit_mid": kwargs.get("take_profit_mid", 0.8),
        "take_profit_far": kwargs.get("take_profit_far", 0.2),
        "time_stop_bars": kwargs.get("time_stop_bars", 30),
        "trend_detection_period": kwargs.get("trend_detection_period", 50),
        "max_trend_slope": kwargs.get("max_trend_slope", 0.002),
        "min_volume_ratio": kwargs.get("min_volume_ratio", 0.8),
        "reversal_confirmation_bars": kwargs.get("reversal_confirmation_bars", 2),
    }
    cerebro.addstrategy(BB_MeanReversion_Strategy, **strategy_params)
    
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / leverage,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    cerebro.broker.set_slippage_perc(0.0001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(DetailedDrawdownAnalyzer, _name="detailed_drawdown")
    cerebro.addanalyzer(TradeRecorder, _name='trade_recorder')
    cerebro.addanalyzer(SQNAnalyzer, _name='sqn')
    
    results = cerebro.run()
    if len(results) > 0:
        strat = results[0][0] if isinstance(results[0], (list, tuple)) else results[0]
    else:
        raise ValueError("No results returned from backtest")

    trades = strat.analyzers.trade_recorder.get_analysis()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else pd.DataFrame()
    loss_trades = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else pd.DataFrame()
    
    winrate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0
    avg_trade = trades_df['pnl'].mean() if not trades_df.empty else 0
    best_trade = trades_df['pnl'].max() if not trades_df.empty else 0
    worst_trade = trades_df['pnl'].min() if not trades_df.empty else 0
    
    max_drawdown = 0
    avg_drawdown = 0
    try:
        dd = strat.analyzers.detailed_drawdown.get_analysis()
        max_drawdown = dd.get('max_drawdown', 0)
        avg_drawdown = dd.get('avg_drawdown', 0)
    except Exception as e:
        print(f"Error accessing drawdown analysis: {e}")

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except Exception:
        sharpe_ratio = 0.0

    profit_factor = (win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())) if not loss_trades.empty else 0
    
    try:
        sqn = strat.analyzers.sqn.get_analysis()['sqn']
    except (AttributeError, KeyError):
        sqn = 0.0

    formatted_results = {
        "Start": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "End": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "Duration": f"{(data.datetime.iloc[-1] - data.datetime.iloc[0]).days} days",
        "Equity Final [$]": final_value,
        "Return [%]": total_return,
        "# Trades": total_trades,
        "Win Rate [%]": winrate,
        "Avg. Trade": avg_trade,
        "Best Trade": best_trade,
        "Worst Trade": worst_trade,
        "Max. Drawdown [%]": max_drawdown,
        "Avg. Drawdown [%]": avg_drawdown,
        "Sharpe Ratio": float(sharpe_ratio),
        "Profit Factor": profit_factor,
        "SQN": sqn,
    }
    
    if verbose:
        print("\n=== Strategy Performance Report ===")
        print(f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${float(formatted_results['Equity Final [$]']):,.2f}")
        print(f"Total Return: {float(formatted_results['Return [%]']):,.2f}%")
        print(f"\nTotal Trades: {int(formatted_results['# Trades'])}")
        print(f"Win Rate: {float(formatted_results['Win Rate [%]']):.2f}%")
        print(f"Best Trade: {float(formatted_results['Best Trade']):.2f}")
        print(f"Worst Trade: {float(formatted_results['Worst Trade']):.2f}")
        print(f"Avg. Trade: {float(formatted_results['Avg. Trade']):.2f}")
        print(f"\nMax Drawdown: {float(formatted_results['Max. Drawdown [%]']):.2f}%")
        print(f"Sharpe Ratio: {float(formatted_results['Sharpe Ratio']):.2f}")
        print(f"Profit Factor: {float(formatted_results['Profit Factor']):.2f}")
    
    return formatted_results

def process_file(args):
    filename, data_folder, leverage = args
    data_path = os.path.join(data_folder, filename)
    
    try:
        parts = filename.split('-')
        symbol = parts[1]
        timeframe = parts[2]
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return (None, filename)

    print(f"\nTesting {symbol} {timeframe} with leverage {leverage}...")
    
    try:
        data_df = pd.read_csv(data_path)
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    except (IOError, ValueError) as e:
        print(f"Error reading or parsing data for {filename}: {str(e)}")
        return (None, filename)

    global LEVERAGE
    LEVERAGE = leverage
    
    results = run_backtest(
        data_df,
        verbose=False,
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        volume_period=20,
        min_range_width=0.01,
        stop_loss_pct=0.005,
        take_profit_mid=0.8,
        take_profit_far=0.2,
        time_stop_bars=30,
        trend_detection_period=50,
        max_trend_slope=0.002,
        min_volume_ratio=0.8,
        reversal_confirmation_bars=2,
        leverage=leverage
    )
    
    log_result(
        strategy="BB_MeanReversion",
        coinpair=symbol,
        timeframe=timeframe,
        leverage=leverage,
        results=results
    )
    
    summary = {
        'symbol': symbol,
        'timeframe': timeframe,
        'leverage': leverage,
        'winrate': results.get('Win Rate [%]', 0),
        'final_equity': results.get('Equity Final [$]', 0),
        'total_trades': results.get('# Trades', 0),
        'max_drawdown': results.get('Max. Drawdown [%]', 0)
    }
    return (summary, filename)

if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    data_folder = os.path.abspath(data_folder)
    
    try:
        files = [f for f in os.listdir(data_folder) if f.startswith('bybit-') and f.endswith('.csv')]
    except (OSError, IOError) as e:
        print(f"Error listing files in {data_folder}: {str(e)}")
        files = []

    all_results = []
    failed_files = []
    leverages = [1, 5, 10, 15, 25, 50]

    try:
        for leverage in leverages:
            print(f"\n==============================\nRunning all backtests for LEVERAGE = {leverage}\n==============================")
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(process_file, [(f, data_folder, leverage) for f in files]))
                
                for summary, fname in results:
                    if summary is not None:
                        all_results.append(summary)
                    else:
                        failed_files.append((fname, leverage))

            # Show top results for this leverage
            leverage_results = [r for r in all_results if r['leverage'] == leverage]
            sorted_results = sorted(leverage_results, key=lambda x: x['winrate'], reverse=True)[:3]
            
            print(f"\n=== Top 3 Results by Win Rate for LEVERAGE {leverage} ===")
            for i, result in enumerate(sorted_results, 1):
                print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                print(f"Win Rate: {result['winrate']:.2f}%")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Final Equity: {result['final_equity']}")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")

        if failed_files:
            print("\nThe following files failed to process:")
            for fname, lev in failed_files:
                print(f"- {fname} (leverage {lev})")

        # if all_results:
        #     pd.DataFrame(all_results).to_csv("results/ema_adx_backtest_results.csv", index=False)

    except Exception as e:
        print("\nException occurred during processing:")
        print(str(e))
        print(traceback.format_exc())
        
        if all_results:
            try:
                # Show partial results for each leverage
                for leverage in leverages:
                    leverage_results = [r for r in all_results if r['leverage'] == leverage]
                    sorted_results = sorted(leverage_results, key=lambda x: x['winrate'], reverse=True)[:3]
                    
                    print(f"\n=== Top 3 Results by Win Rate (Partial) for LEVERAGE {leverage} ===")
                    for i, result in enumerate(sorted_results, 1):
                        print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                        print(f"Win Rate: {result['winrate']:.2f}%")
                        print(f"Total Trades: {result['total_trades']}")
                        print(f"Final Equity: {result['final_equity']}")
                        print(f"Max Drawdown: {result['max_drawdown']:.2f}%")

                if failed_files:
                    print("\nThe following files failed to process:")
                    for fname, lev in failed_files:
                        print(f"- {fname} (leverage {lev})")

                if all_results:
                    pd.DataFrame(all_results).to_csv("ema_adx_backtest_results.csv", index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc())
