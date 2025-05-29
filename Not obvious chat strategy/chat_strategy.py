import backtrader as bt
import pandas as pd
from datetime import datetime
import math
import traceback
from typing import Optional
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer
from results_logger import log_result


LEVERAGE = 50

class VwapIntradayIndicator(bt.Indicator):
    """
    Volume Weighted Average Price (VWAP) indicator for intraday trading.
    Resets on a new day based on the date from the datetime field.
    """
    
    lines = ("vwap_intraday",)
    params = {}
    plotinfo = {"subplot": False}
    plotlines = {"vwap_intraday": {"color": "blue"}}

    def __init__(self) -> None:
        # Calculate the typical price
        self.hlc = (self.data.high + self.data.low + self.data.close) / 3.0

        # Initialize tracking variables
        self.current_date: Optional[datetime.date] = None
        self.previous_date_index: int = -1
        
        # Initialize cumulative values
        self.cum_vol = 0.0
        self.cum_hlc_vol = 0.0

    def next(self) -> None:
        try:
            # Extract the current date from the datetime field
            current_datetime = self.data.datetime.datetime()
            current_date = current_datetime.date()

            # Check if the date has changed
            if self.current_date != current_date:
                # Reset cumulative values on new day
                self.current_date = current_date
                self.cum_vol = 0.0
                self.cum_hlc_vol = 0.0

            # Get current values
            current_volume = self.data.volume[0]
            current_hlc = self.hlc[0]

            # Update cumulative values
            if not math.isnan(current_volume) and not math.isnan(current_hlc):
                self.cum_vol += current_volume
                self.cum_hlc_vol += current_hlc * current_volume

            # Calculate VWAP with safety check for zero volume
            if self.cum_vol > 0:
                self.lines.vwap_intraday[0] = self.cum_hlc_vol / self.cum_vol
            else:
                # If no volume, use the typical price as fallback
                self.lines.vwap_intraday[0] = current_hlc

        except ZeroDivisionError:
            print("ZeroDivisionError in VWAP calculation: cum_vol was zero.")
            self.lines.vwap_intraday[0] = self.hlc[0]
        except Exception as e:
            print(f"Error in VWAP calculation: {str(e)}")
            # Use typical price as fallback in case of any error
            self.lines.vwap_intraday[0] = self.hlc[0]

class KeltnerChannels(bt.Indicator):
    """Keltner Channels indicator for Backtrader"""
    
    lines = ('kc_upper', 'kc_middle', 'kc_lower',)
    params = (
        ('period', 20),
        ('multiplier', 2.0),
        ('atr_period', 10),
    )

    def __init__(self):
        # Calculate the middle band using EMA
        self.lines.kc_middle = bt.indicators.EMA(self.data.close, period=self.p.period)
        
        # Calculate ATR for volatility
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # Calculate upper and lower bands with zero-division protection
        self.lines.kc_upper = self.lines.kc_middle + self.p.multiplier * self.atr
        self.lines.kc_lower = self.lines.kc_middle - self.p.multiplier * self.atr

class SafeStochRSI(bt.Indicator):
    """
    Fully self-contained Stochastic RSI:
      1) computes a Wilder RSI in next()
      2) then computes StochRSI on that RSI series
      3) applies K and D smoothing
      4) exposes rsi, stochrsi_k and stochrsi_d lines
    Never divides by zero.
    """
    lines = ('stochrsi_k', 'stochrsi_d', 'rsi',)
    params = (
        ('period',    14),  # lookback for RSI *and* StochRSI
        ('smooth_k',   3),
        ('smooth_d',   3),
    )

    def __init__(self):
        # buffers for close-price, RSI, raw K and D
        self._price_buf = []
        self._rsi_buf   = []
        self._k_buf     = []
        self._d_buf     = []

        # need enough bars to fill:
        #   period+1 closes → RSI,
        #   then period RSI → raw %K,
        #   plus smooth_k + smooth_d for smoothing
        minbars = self.p.period + 1 + self.p.period + self.p.smooth_k + self.p.smooth_d
        self.addminperiod(minbars)

    def next(self):
        close = self.data.close[0]

        # 1) price buffer for RSI
        self._price_buf.append(close)
        if len(self._price_buf) > self.p.period + 1:
            self._price_buf.pop(0)

        # 2) compute Wilder's RSI
        if len(self._price_buf) == self.p.period + 1:
            gains = losses = 0.0
            for i in range(1, len(self._price_buf)):
                diff = self._price_buf[i] - self._price_buf[i-1]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            avg_gain = gains / self.p.period
            avg_loss = losses / self.p.period

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs  = avg_gain / (avg_loss + 1e-8)
                rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0  # neutral until enough data

        # 3) buffer for StochRSI
        self._rsi_buf.append(rsi)
        if len(self._rsi_buf) > self.p.period:
            self._rsi_buf.pop(0)

        # 4) raw %K
        if len(self._rsi_buf) == self.p.period:
            rmin, rmax = min(self._rsi_buf), max(self._rsi_buf)
            if rmax > rmin:
                raw_k = (rsi - rmin) / (rmax - rmin) * 100.0
            else:
                raw_k = 0.0
        else:
            raw_k = 50.0

        # 5) smooth %K
        self._k_buf.append(raw_k)
        if len(self._k_buf) > self.p.smooth_k:
            self._k_buf.pop(0)
        k_val = sum(self._k_buf) / len(self._k_buf)

        # 6) smooth %D
        self._d_buf.append(k_val)
        if len(self._d_buf) > self.p.smooth_d:
            self._d_buf.pop(0)
        d_val = sum(self._d_buf) / len(self._d_buf)

        # 7) write out all three lines
        self.lines.rsi[0]           = rsi
        self.lines.stochrsi_k[0]    = k_val
        self.lines.stochrsi_d[0]    = d_val





class ChatStrategy(bt.Strategy):
    """Base template for creating trading strategies"""
    
    params = (
        ("keltner_period", 20),
        ("keltner_multiplier", 2.0),
        ("stochrsi_period", 14),
        ("stochrsi_k", 3),
        ("stochrsi_d", 3),
        ("stop_loss_pct", 0.01),
        ("take_profit_ratio", 0.01),
    )


    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []  # To track ongoing trades for visualization
        
        self.vwap_intraday = VwapIntradayIndicator()
        self.keltner = KeltnerChannels(
            period=self.p.keltner_period,
            multiplier=self.p.keltner_multiplier
        )
        self.stochrsi = SafeStochRSI(
            period=self.p.stochrsi_period,
            smooth_k=self.p.stochrsi_k,
            smooth_d=self.p.stochrsi_d
)

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()
            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0
            leverage = LEVERAGE
            # Adjust position size according to leverage
            if current_price == 0:
                return 0
            position_size = (position_value * leverage) / current_price
            return position_size
        except ZeroDivisionError:
            print("ZeroDivisionError in calculate_position_size: current_price was zero.")
            return 0
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def next(self):
        """Define trading logic"""
        # Skip if not enough data for indicators
        if (math.isnan(self.vwap_intraday[0]) or math.isinf(self.vwap_intraday[0]) or
            math.isnan(self.keltner.kc_lower[0]) or math.isinf(self.keltner.kc_lower[0]) or
            math.isnan(self.keltner.kc_upper[0]) or math.isinf(self.keltner.kc_upper[0]) or
            math.isnan(self.stochrsi.stochrsi_k[0]) or math.isinf(self.stochrsi.stochrsi_k[0]) or
            math.isnan(self.stochrsi.stochrsi_d[0]) or math.isinf(self.stochrsi.stochrsi_d[0])):
            return

        # Get current position
        position = self.position.size
        
        # Get current price and indicator values
        current_price = self.data.close[0]
        keltner_lower = self.keltner.kc_lower[0]
        keltner_upper = self.keltner.kc_upper[0]
        keltner_middle = self.keltner.kc_middle[0]
        vwap = self.vwap_intraday[0]
        stochrsi_k = self.stochrsi.stochrsi_k[0]
        stochrsi_d = self.stochrsi.stochrsi_d[0]
        rsi = self.stochrsi.rsi[0]

        # Exit conditions for existing positions
        if position:
            # For long positions
            if position > 0:
                # Take profit at VWAP or middle Keltner band
                if current_price >= min(vwap, keltner_middle) or \
                   (current_price - self.position.price) / self.position.price >= self.p.take_profit_ratio:
                    self.close()
                # Stop loss
                elif (current_price - self.position.price) / self.position.price <= -self.p.stop_loss_pct:
                    self.close()
            # For short positions
            elif position < 0:
                # Take profit at VWAP or middle Keltner band
                if current_price <= max(vwap, keltner_middle) or \
                   (self.position.price - current_price) / self.position.price >= self.p.take_profit_ratio:
                    self.close()
                # Stop loss
                elif (self.position.price - current_price) / self.position.price <= -self.p.stop_loss_pct:
                    self.close()
            # Do not check entry logic if a position is open
            return

        # Entry conditions (only if no position is open)
        # Long signal when StochRSI K crosses above 20 from oversold (wider window)
        long_signal = (
            current_price < keltner_lower and
            current_price < vwap and
            self.stochrsi.stochrsi_k[-2] < 20 and  # Two bars ago K value
            stochrsi_k > 20  # Current K value
        )

        # Short signal when StochRSI K crosses below 80 from overbought (wider window)
        short_signal = (
            current_price > keltner_upper and
            current_price > vwap and
            self.stochrsi.stochrsi_k[-2] > 80 and  # Two bars ago K value
            stochrsi_k < 80  # Current K value
        )

        # Calculate position size only if we need to enter a trade
        # Execute trades
        if long_signal:
            position_size = self.calculate_position_size(current_price)
            stopprice = current_price - (current_price * self.p.stop_loss_pct)
            limitprice = current_price * (1 + self.p.take_profit_ratio)
            self.buy_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                limitprice=limitprice,
                stopprice=stopprice
            )
            # self.log("Long Position Entered")

        elif short_signal:
            position_size = self.calculate_position_size(current_price)
            stopprice = current_price + (current_price * self.p.stop_loss_pct)
            limitprice = current_price * (1 - self.p.take_profit_ratio)
            self.sell_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                limitprice=limitprice,
                stopprice=stopprice
            )
            # self.log("Short Position Entered")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        try:
            # Get entry and exit prices
            entry_price = trade.price
            exit_price = trade.history[-1].price if trade.history else self.data.close[0]
            pnl = trade.pnl
            
            # Store trade exit information for visualization
            self.trade_exits.append({
                'datetime': self.data.datetime.datetime(0),
                'price': exit_price,
                'type': 'long_exit' if trade.size > 0 else 'short_exit',
                'pnl': pnl,
                'entry_price': entry_price
            })
            
        except Exception as e:
            print(f"Warning: Could not process trade: {str(e)}")
            print(f"Trade info - Status: {trade.status}, Size: {trade.size}, "
                  f"Price: {trade.price}, PnL: {trade.pnl}")

    def notify_order(self, order):
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

def run_backtest(data, verbose=True, **kwargs):
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
        "keltner_period": kwargs.get("keltner_period", 20),
        "keltner_multiplier": kwargs.get("keltner_multiplier", 2.0),
        "stochrsi_period": kwargs.get("stochrsi_period", 14),
        "stochrsi_k": kwargs.get("stochrsi_k", 3),
        "stochrsi_d": kwargs.get("stochrsi_d", 3),
        "stop_loss_pct": kwargs.get("stop_loss_pct", 0.01),
        "take_profit_ratio": kwargs.get("take_profit_ratio", 0.01),
    }
    cerebro.addstrategy(ChatStrategy, **strategy_params)
    initial_cash = 100.0
    leverage = LEVERAGE
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
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame()
    total_trades = len(trades_df)
    if not trades_df.empty:
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] < 0]
        winrate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_trade = trades_df['pnl'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
    else:
        win_trades = pd.DataFrame()
        loss_trades = pd.DataFrame()
        winrate = 0
        avg_trade = 0
        best_trade = 0
        worst_trade = 0
    max_drawdown = 0
    avg_drawdown = 0
    try:
        dd = strat.analyzers.detailed_drawdown.get_analysis()
        max_drawdown = dd.get('max_drawdown', 0)
        avg_drawdown = dd.get('avg_drawdown', 0)
    except (AttributeError, KeyError) as e:
        print(f"Error accessing drawdown analysis: {e}")
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except (AttributeError, KeyError):
        sharpe_ratio = 0.0
    profit_factor = (win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())) if not loss_trades.empty else 0
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
        keltner_period=20,
        keltner_multiplier=2.0,
        stochrsi_period=14,
        stochrsi_k=3,
        stochrsi_d=3,
        stop_loss_pct=0.01,
        take_profit_ratio=0.01
    )
    log_result(
            strategy="ChatStrategy",
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
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
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
            # Print top 3 for this leverage
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
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred during processing:")
        print(str(e))
        print(traceback.format_exc())
        if all_results:
            try:
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
                pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc()) 