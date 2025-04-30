import backtrader as bt
import numpy as np
import math

class TradeRecorder(bt.Analyzer):
    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.active_trades = {}
        self.trades = []
        self.bar_counter = 0
    def next(self):
        self.bar_counter += 1
    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            trade_value = abs(trade.price * trade.size)
            self.active_trades[trade.ref] = {
                'entry_time': self.bar_counter,
                'entry_bar_datetime': bt.num2date(self.datas[0].datetime[0]),
                'entry_price': trade.price,
                'size': abs(trade.size),
                'value': trade_value
            }
        if trade.isclosed:
            entry_data = self.active_trades.pop(trade.ref, None)
            if entry_data is not None:
                entry_time = entry_data['entry_time']
                exit_time = self.bar_counter
                bars_held = exit_time - entry_time
                exit_price = trade.price
                self.trades.append({
                    'datetime': bt.num2date(self.datas[0].datetime[0]),
                    'type': 'long' if trade.size > 0 else 'short',
                    'size': entry_data['size'],
                    'price': exit_price,
                    'value': entry_data['value'],
                    'pnl': float(trade.pnl),
                    'pnlcomm': float(trade.pnlcomm),
                    'commission': float(trade.commission),
                    'entry_price': entry_data['entry_price'],
                    'exit_price': exit_price,
                    'bars_held': bars_held
                })
    def get_analysis(self):
        return self.trades

class SQNAnalyzer(bt.Analyzer):
    def __init__(self):
        super(SQNAnalyzer, self).__init__()
        self.trades = []
        self.sqn = 0.0
    def start(self):
        self.trades = []
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append(float(trade.pnl))
    def stop(self):
        if len(self.trades) < 2:
            self.sqn = 0.0
        else:
            pnl_list = self.trades
            avg_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list)
            if std_pnl == 0:
                self.sqn = 0.0
            else:
                self.sqn = (avg_pnl / std_pnl) * math.sqrt(len(pnl_list))
    def get_analysis(self):
        return {'sqn': self.sqn}

class DetailedDrawdownAnalyzer(bt.Analyzer):
    def __init__(self):
        super(DetailedDrawdownAnalyzer, self).__init__()
        self.drawdowns = []
        self.current_drawdown = None
        self.peak = 0
        self.equity_curve = []
        self.bar_counter = 0
        self.max_drawdown = 0
        self.max_drawdown_start = None
        self.max_drawdown_end = None
    def next(self):
        value = self.strategy.broker.getvalue()
        self.equity_curve.append(value)
        if value > self.peak:
            self.peak = value
            if self.current_drawdown is not None:
                self.drawdowns.append(self.current_drawdown)
                self.current_drawdown = None
        elif value < self.peak:
            dd_pct = (self.peak - value) / self.peak * 100
            if self.current_drawdown is None:
                self.current_drawdown = {'start': self.bar_counter, 'peak': self.peak, 'lowest': value, 'dd_pct': dd_pct}
            elif value < self.current_drawdown['lowest']:
                self.current_drawdown['lowest'] = value
                self.current_drawdown['dd_pct'] = dd_pct
            if dd_pct > self.max_drawdown:
                self.max_drawdown = dd_pct
                self.max_drawdown_start = self.current_drawdown['start'] if self.current_drawdown else None
                self.max_drawdown_end = self.bar_counter
        self.bar_counter += 1
    def stop(self):
        if self.current_drawdown is not None:
            self.drawdowns.append(self.current_drawdown)
    def get_analysis(self):
        avg_dd = sum(dd['dd_pct'] for dd in self.drawdowns) / len(self.drawdowns) if self.drawdowns else 0
        return {
            'drawdowns': self.drawdowns,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_start': self.max_drawdown_start,
            'max_drawdown_end': self.max_drawdown_end,
            'avg_drawdown': avg_dd,
            'equity_curve': self.equity_curve
        } 