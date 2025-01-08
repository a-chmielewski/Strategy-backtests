import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    def __init__(self, data_paths):
        """
        Initialize the MarketAnalyzer with paths to data files
        data_paths: dict with keys as (symbol, timeframe) tuples and values as file paths
        """
        self.data = {}
        self.interpretations = {}
        for (symbol, timeframe), path in data_paths.items():
            df = pd.read_csv(path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            self.data[(symbol, timeframe)] = df
            
    def interpret_volatility(self, symbol, timeframe):
        """Interpret volatility metrics and patterns"""
        df = self.calculate_volatility_metrics(symbol, timeframe)
        
        # Calculate key metrics
        recent_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].mean()
        atr_percentile = stats.percentileofscore(df['ATR'].dropna(), recent_atr)
        
        recent_vol_1h = df['rolling_vol_1h'].iloc[-1]
        avg_vol_1h = df['rolling_vol_1h'].mean()
        vol_1h_percentile = stats.percentileofscore(df['rolling_vol_1h'].dropna(), recent_vol_1h)
        
        # Volatility clustering analysis
        vol_autocorr = df['rolling_vol_1h'].autocorr()
        
        interpretation = {
            'current_market_state': {
                'description': self._interpret_current_volatility(recent_atr, avg_atr, atr_percentile),
                'metrics': {
                    'recent_atr': recent_atr,
                    'avg_atr': avg_atr,
                    'atr_percentile': atr_percentile,
                    'recent_vol_1h': recent_vol_1h,
                    'avg_vol_1h': avg_vol_1h,
                    'vol_1h_percentile': vol_1h_percentile
                }
            },
            'volatility_clustering': {
                'description': self._interpret_volatility_clustering(vol_autocorr),
                'autocorrelation': vol_autocorr
            },
            'trading_implications': self._get_volatility_trading_implications(recent_atr, avg_atr, vol_autocorr)
        }
        
        return interpretation
    
    def interpret_returns(self, symbol, timeframe):
        """Interpret returns distribution and patterns"""
        df = self.data[(symbol, timeframe)].copy()
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate key statistics
        returns_stats = {
            'mean': df['Returns'].mean(),
            'std': df['Returns'].std(),
            'skew': df['Returns'].skew(),
            'kurtosis': df['Returns'].kurtosis(),
            'sharpe': (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(len(df)),
            'var_95': np.percentile(df['Returns'].dropna(), 5),
            'var_99': np.percentile(df['Returns'].dropna(), 1)
        }
        
        # Normality test
        _, normality_pvalue = stats.normaltest(df['Returns'].dropna())
        
        interpretation = {
            'distribution_characteristics': {
                'description': self._interpret_returns_distribution(returns_stats),
                'metrics': returns_stats
            },
            'risk_metrics': {
                'description': self._interpret_risk_metrics(returns_stats),
                'var_95': returns_stats['var_95'],
                'var_99': returns_stats['var_99']
            },
            'normality': {
                'description': self._interpret_normality(normality_pvalue),
                'p_value': normality_pvalue
            },
            'trading_implications': self._get_returns_trading_implications(returns_stats)
        }
        
        return interpretation
    
    def interpret_temporal_patterns(self, symbol, timeframe):
        """Interpret temporal patterns in the data"""
        df = self.data[(symbol, timeframe)].copy()
        df['Returns'] = df['Close'].pct_change()
        df['abs_Returns'] = abs(df['Returns'])
        
        # Hourly analysis
        hourly_stats = df.groupby(df.index.hour).agg({
            'abs_Returns': ['mean', 'std'],
            'Volume': ['mean', 'std']
        })
        
        # Daily analysis
        daily_stats = df.groupby(df.index.day_name()).agg({
            'abs_Returns': ['mean', 'std'],
            'Volume': ['mean', 'std']
        })
        
        # Find best trading hours
        best_hours = hourly_stats['abs_Returns']['mean'].nlargest(3)
        worst_hours = hourly_stats['abs_Returns']['mean'].nsmallest(3)
        
        # Find best trading days
        best_days = daily_stats['abs_Returns']['mean'].nlargest(3)
        worst_days = daily_stats['abs_Returns']['mean'].nsmallest(3)
        
        interpretation = {
            'hourly_patterns': {
                'description': self._interpret_hourly_patterns(best_hours, worst_hours),
                'best_hours': best_hours.index.tolist(),
                'worst_hours': worst_hours.index.tolist()
            },
            'daily_patterns': {
                'description': self._interpret_daily_patterns(best_days, worst_days),
                'best_days': best_days.index.tolist(),
                'worst_days': worst_days.index.tolist()
            },
            'trading_implications': self._get_temporal_trading_implications(best_hours, best_days)
        }
        
        return interpretation
    
    def generate_comprehensive_report(self, symbol, timeframe):
        """Generate a comprehensive analysis report"""
        volatility_interpretation = self.interpret_volatility(symbol, timeframe)
        returns_interpretation = self.interpret_returns(symbol, timeframe)
        temporal_interpretation = self.interpret_temporal_patterns(symbol, timeframe)
        
        report = f"""
Market Analysis Report for {symbol} ({timeframe})
================================================

1. Current Market State
----------------------
{volatility_interpretation['current_market_state']['description']}

2. Volatility Analysis
---------------------
- Market Volatility: {volatility_interpretation['current_market_state']['description']}
- Volatility Clustering: {volatility_interpretation['volatility_clustering']['description']}
- Trading Implications: {volatility_interpretation['trading_implications']}

3. Returns Analysis
------------------
- Distribution Characteristics: {returns_interpretation['distribution_characteristics']['description']}
- Risk Metrics: {returns_interpretation['risk_metrics']['description']}
- Normality Test: {returns_interpretation['normality']['description']}
- Trading Implications: {returns_interpretation['trading_implications']}

4. Temporal Patterns
-------------------
- Hourly Patterns: {temporal_interpretation['hourly_patterns']['description']}
- Daily Patterns: {temporal_interpretation['daily_patterns']['description']}
- Trading Implications: {temporal_interpretation['trading_implications']}

5. Recommended Scalping Strategy Parameters
----------------------------------------
{self._generate_strategy_recommendations(symbol, timeframe, volatility_interpretation, returns_interpretation, temporal_interpretation)}
"""
        return report
    
    def _interpret_current_volatility(self, recent_atr, avg_atr, atr_percentile):
        """Interpret current volatility state"""
        if atr_percentile > 80:
            state = "extremely high"
        elif atr_percentile > 60:
            state = "above average"
        elif atr_percentile > 40:
            state = "average"
        elif atr_percentile > 20:
            state = "below average"
        else:
            state = "extremely low"
            
        return f"Current volatility is {state} (ATR percentile: {atr_percentile:.1f}%). " \
               f"Recent ATR: {recent_atr:.8f}, Average ATR: {avg_atr:.8f}"
    
    def _interpret_volatility_clustering(self, autocorr):
        """Interpret volatility clustering"""
        if autocorr > 0.7:
            return "Strong volatility clustering present - high volatility periods likely to persist"
        elif autocorr > 0.3:
            return "Moderate volatility clustering - some predictability in volatility patterns"
        else:
            return "Weak volatility clustering - volatility patterns show limited persistence"
    
    def _interpret_returns_distribution(self, stats):
        """Interpret returns distribution characteristics"""
        description = []
        
        # Interpret skewness
        if abs(stats['skew']) < 0.2:
            description.append("Returns are roughly symmetrical")
        elif stats['skew'] > 0:
            description.append("Returns show positive skew (more extreme positive returns)")
        else:
            description.append("Returns show negative skew (more extreme negative returns)")
        
        # Interpret kurtosis
        if stats['kurtosis'] > 3:
            description.append("with heavy tails (more extreme events than normal)")
        elif stats['kurtosis'] < 3:
            description.append("with light tails (fewer extreme events than normal)")
        
        return " ".join(description)
    
    def _interpret_risk_metrics(self, stats):
        """Interpret risk metrics"""
        var95_pct = stats['var_95'] * 100
        var99_pct = stats['var_99'] * 100
        
        return f"95% VaR: {var95_pct:.2f}% (95% confidence of not losing more than this in a single period)\n" \
               f"99% VaR: {var99_pct:.2f}% (99% confidence of not losing more than this in a single period)"
    
    def _interpret_normality(self, p_value):
        """Interpret normality test results"""
        if p_value < 0.05:
            return "Returns are not normally distributed (p < 0.05)"
        else:
            return "Returns appear to be normally distributed"
    
    def _interpret_hourly_patterns(self, best_hours, worst_hours):
        """Interpret hourly patterns"""
        return f"Best trading hours: {', '.join(map(str, best_hours.index))} UTC\n" \
               f"Worst trading hours: {', '.join(map(str, worst_hours.index))} UTC"
    
    def _interpret_daily_patterns(self, best_days, worst_days):
        """Interpret daily patterns"""
        return f"Best trading days: {', '.join(best_days.index)}\n" \
               f"Worst trading days: {', '.join(worst_days.index)}"
    
    def _get_volatility_trading_implications(self, recent_atr, avg_atr, vol_autocorr):
        """Generate trading implications based on volatility analysis"""
        implications = []
        
        # ATR-based implications
        atr_ratio = recent_atr / avg_atr
        if atr_ratio > 1.2:
            implications.append("Consider wider stop-losses due to higher volatility")
        elif atr_ratio < 0.8:
            implications.append("Tighter stop-losses may be appropriate in current conditions")
            
        # Volatility clustering implications
        if vol_autocorr > 0.5:
            implications.append("Strong volatility persistence suggests maintaining consistent position sizing")
        
        return " ".join(implications)
    
    def _get_returns_trading_implications(self, stats):
        """Generate trading implications based on returns analysis"""
        implications = []
        
        # Sharpe ratio implications
        if stats['sharpe'] > 1:
            implications.append("Favorable risk-adjusted returns suggest aggressive position sizing")
        elif stats['sharpe'] < 0:
            implications.append("Poor risk-adjusted returns suggest defensive position sizing")
            
        # Tail risk implications
        if stats['kurtosis'] > 5:
            implications.append("Heavy tails suggest implementing strict stop-losses")
        
        return " ".join(implications)
    
    def _get_temporal_trading_implications(self, best_hours, best_days):
        """Generate trading implications based on temporal analysis"""
        return f"Focus trading during peak hours: {', '.join(map(str, best_hours.index))} UTC, " \
               f"particularly on {', '.join(best_days.index)}"
    
    def _generate_strategy_recommendations(self, symbol, timeframe, vol_interp, ret_interp, temp_interp):
        """Generate specific strategy recommendations"""
        recent_atr = vol_interp['current_market_state']['metrics']['recent_atr']
        
        recommendations = f"""
Based on the analysis, here are the recommended parameters for a scalping strategy:

1. Position Sizing:
   - Base position size: 1-2% of trading capital
   - Adjust based on current volatility state: {vol_interp['current_market_state']['description']}

2. Stop Loss and Take Profit:
   - Initial stop loss: {recent_atr * 1.5:.8f} (1.5 * ATR)
   - Take profit: {recent_atr * 2:.8f} (2 * ATR)
   - Consider using trailing stops of {recent_atr:.8f} (1 * ATR)

3. Optimal Trading Windows:
   - Primary hours: {', '.join(map(str, temp_interp['hourly_patterns']['best_hours']))} UTC
   - Best days: {', '.join(temp_interp['daily_patterns']['best_days'])}

4. Risk Management:
   - Maximum daily loss: {abs(ret_interp['risk_metrics']['var_99'] * 100 * 3):.2f}% of trading capital
   - Maximum position loss: {abs(ret_interp['risk_metrics']['var_95'] * 100):.2f}% per trade

5. Market Conditions:
   - Volatility State: {vol_interp['current_market_state']['description']}
   - Returns Profile: {ret_interp['distribution_characteristics']['description']}
"""
        return recommendations
            
    def plot_ohlc(self, symbol, timeframe):
        """Plot OHLC chart with volume"""
        df = self.data[(symbol, timeframe)]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=(f'{symbol} OHLC', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='OHLC'),
                     row=1, col=1)
        
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                            name='Volume'),
                     row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} Price and Volume Analysis ({timeframe})',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def calculate_volatility_metrics(self, symbol, timeframe):
        """Calculate various volatility metrics"""
        df = self.data[(symbol, timeframe)].copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Calculate rolling volatility
        df['rolling_vol_1h'] = df['Returns'].rolling(window=60).std() * np.sqrt(60)
        df['rolling_vol_4h'] = df['Returns'].rolling(window=240).std() * np.sqrt(240)
        
        return df
    
    def plot_volatility_analysis(self, symbol, timeframe):
        """Plot volatility analysis"""
        df = self.calculate_volatility_metrics(symbol, timeframe)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Price with ATR', 'Rolling Volatility'))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                name='Close Price'),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'],
                                name='ATR'),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['rolling_vol_1h'],
                                name='1h Rolling Vol'),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['rolling_vol_4h'],
                                name='4h Rolling Vol'),
                     row=2, col=1)
        
        fig.update_layout(title=f'{symbol} Volatility Analysis ({timeframe})')
        return fig
    
    def analyze_returns_distribution(self, symbol, timeframe):
        """Analyze and plot returns distribution"""
        df = self.data[(symbol, timeframe)].copy()
        df['Returns'] = df['Close'].pct_change()
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Returns Distribution', 'Q-Q Plot',
                                         'Returns Over Time', 'Box Plot'))
        
        # Returns distribution
        returns_hist = go.Histogram(x=df['Returns'].dropna(),
                                  name='Returns Distribution',
                                  nbinsx=50)
        fig.add_trace(returns_hist, row=1, col=1)
        
        # Q-Q plot
        qq = stats.probplot(df['Returns'].dropna(), dist="norm")
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1],
                                mode='markers',
                                name='Q-Q Plot'),
                     row=1, col=2)
        
        # Returns over time
        fig.add_trace(go.Scatter(x=df.index, y=df['Returns'],
                                mode='lines',
                                name='Returns'),
                     row=2, col=1)
        
        # Box plot
        fig.add_trace(go.Box(y=df['Returns'].dropna(),
                            name='Returns Distribution'),
                     row=2, col=2)
        
        fig.update_layout(title=f'{symbol} Returns Analysis ({timeframe})')
        return fig
    
    def analyze_temporal_patterns(self, symbol, timeframe):
        """Analyze temporal patterns in the data"""
        df = self.data[(symbol, timeframe)].copy()
        df['Returns'] = df['Close'].pct_change()
        df['abs_Returns'] = abs(df['Returns'])
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.day_name()
        
        # Hourly analysis
        hourly_volatility = df.groupby('hour')['abs_Returns'].mean()
        hourly_volume = df.groupby('hour')['Volume'].mean()
        
        # Daily analysis
        daily_volatility = df.groupby('day_of_week')['abs_Returns'].mean()
        daily_volume = df.groupby('day_of_week')['Volume'].mean()
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Hourly Volatility', 'Hourly Volume',
                                         'Daily Volatility', 'Daily Volume'))
        
        # Hourly plots
        fig.add_trace(go.Bar(x=hourly_volatility.index,
                            y=hourly_volatility.values,
                            name='Hourly Volatility'),
                     row=1, col=1)
        
        fig.add_trace(go.Bar(x=hourly_volume.index,
                            y=hourly_volume.values,
                            name='Hourly Volume'),
                     row=1, col=2)
        
        # Daily plots
        fig.add_trace(go.Bar(x=daily_volatility.index,
                            y=daily_volatility.values,
                            name='Daily Volatility'),
                     row=2, col=1)
        
        fig.add_trace(go.Bar(x=daily_volume.index,
                            y=daily_volume.values,
                            name='Daily Volume'),
                     row=2, col=2)
        
        fig.update_layout(title=f'{symbol} Temporal Patterns Analysis ({timeframe})')
        return fig

def main():
    # Define data paths
    data_paths = {
        ('BTC/USDT', '1m'): r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20240929-to-20241128.csv',
        ('BTC/USDT', '5m'): r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv',
        ('ETH/USDT', '1m'): r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-1m-20240929-to-20241128.csv',
        ('ETH/USDT', '5m'): r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-5m-20240929-to-20241128.csv'
    }
    
    # Initialize analyzer
    analyzer = MarketAnalyzer(data_paths)
    
    # Generate and save all analyses
    for (symbol, timeframe) in data_paths.keys():
        # Generate visual analyses
        fig_ohlc = analyzer.plot_ohlc(symbol, timeframe)
        fig_ohlc.write_html(f'analysis_{symbol.replace("/", "")}_{timeframe}_ohlc.html')
        
        fig_vol = analyzer.plot_volatility_analysis(symbol, timeframe)
        fig_vol.write_html(f'analysis_{symbol.replace("/", "")}_{timeframe}_volatility.html')
        
        fig_ret = analyzer.analyze_returns_distribution(symbol, timeframe)
        fig_ret.write_html(f'analysis_{symbol.replace("/", "")}_{timeframe}_returns.html')
        
        fig_temp = analyzer.analyze_temporal_patterns(symbol, timeframe)
        fig_temp.write_html(f'analysis_{symbol.replace("/", "")}_{timeframe}_temporal.html')
        
        # Generate and save interpretation report
        report = analyzer.generate_comprehensive_report(symbol, timeframe)
        with open(f'analysis_{symbol.replace("/", "")}_{timeframe}_report.txt', 'w') as f:
            f.write(report)

if __name__ == "__main__":
    main()