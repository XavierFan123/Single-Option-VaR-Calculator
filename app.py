# /option_var_app/app.py

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止在服务器上出错
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import io
import base64

from flask import Flask, render_template, request, jsonify

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 从Jupyter Notebook移植过来的核心计算类 ---
# (为了简洁，我将所有类都放在这个文件里)

class TreasuryRateFetcher:
    YIELD_TICKERS = {
        '^IRX': 0.25,  # 3个月
        '^FVX': 5,     # 5年
        '^TNX': 10,    # 10年
        '^TYX': 30,    # 30年
    }
    
    @staticmethod
    def get_yield_curve():
        rates = {}
        for ticker, maturity in TreasuryRateFetcher.YIELD_TICKERS.items():
            try:
                bond = yf.Ticker(ticker)
                hist = bond.history(period="5d")
                if not hist.empty:
                    current_yield = hist['Close'].iloc[-1] / 100
                    rates[maturity] = current_yield
            except Exception:
                continue
        # 如果获取失败，提供一个备用默认值（优化：更完整的默认曲线）
        if not rates:
            return {0.25: 0.045, 1: 0.046, 2: 0.047, 5: 0.048, 10: 0.049, 30: 0.050}
        return rates

    @staticmethod
    def get_risk_free_rate(expiry_date_str):
        try:
            expiry = pd.Timestamp(expiry_date_str)
            current = pd.Timestamp.now()
            years_to_expiry = (expiry - current).days / 365.0
            
            rates = TreasuryRateFetcher.get_yield_curve()
            
            if rates:
                maturities = sorted(rates.keys())
                yields = [rates[m] for m in maturities]
                
                if years_to_expiry <= maturities[0]: return yields[0]
                if years_to_expiry >= maturities[-1]: return yields[-1]
                return np.interp(years_to_expiry, maturities, yields)
            return 0.05  # 默认值
        except Exception:
            return 0.05  # 默认值

class ImpliedVolatilityCalculator:
    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def calculate_iv(option_price, S, K, T, r, option_type='call', chain=None):
        if T <= 0: return np.nan
        objective = lambda sigma: ImpliedVolatilityCalculator.black_scholes_price(S, K, T, r, sigma, option_type) - option_price
        try:
            return brentq(objective, 0.001, 5.0, maxiter=100)
        except (ValueError, RuntimeError):
            # 优化：添加备用IV逻辑
            if chain is not None and 'impliedVolatility' in chain.columns:
                fallback_iv = chain['impliedVolatility'].iloc[0]
                if not np.isnan(fallback_iv):
                    return fallback_iv
            # 进一步备用：计算历史波动率（基于60天历史数据）
            try:
                stock = yf.Ticker(chain['contractSymbol'].iloc[0].split('_')[0])
                hist = stock.history(period="3mo")
                if len(hist) >= 2:
                    returns = np.log(hist['Close'] / hist['Close'].shift(1))
                    hist_vol = returns.std() * np.sqrt(252)  # 年化
                    return hist_vol
            except:
                pass
            return np.nan  # 如果所有失败，返回nan

class GreeksCalculator:
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        greeks = {}
        if option_type.lower() == 'call':
            greeks['delta'] = norm.cdf(d1)
            greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
            greeks['rho'] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # Put
            greeks['delta'] = norm.cdf(d1) - 1
            greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
            greeks['rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return greeks

class MonteCarloEngine:
    @staticmethod
    def simulate_price_paths(S0, r, sigma, T, time_horizon_days, n_simulations):
        dt = time_horizon_days / 365.0
        Z = np.random.standard_normal(n_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return ST

    @staticmethod
    def calculate_option_values(simulated_prices, K, T, r, sigma, option_type):
        return ImpliedVolatilityCalculator.black_scholes_price(simulated_prices, K, T, r, sigma, option_type)

class RiskCalculator:
    @staticmethod
    def calculate_time_to_expiry(expiry_date_str):
        return (pd.Timestamp(expiry_date_str) - pd.Timestamp.now()).days / 365.0

    @staticmethod
    def calculate_var(pnl, confidence_level):
        # 优化：使用'lower'方法，更保守的分位计算
        return np.percentile(pnl, (1 - confidence_level) * 100, method='lower')

    @staticmethod
    def calculate_cvar(pnl, confidence_level):
        var = RiskCalculator.calculate_var(pnl, confidence_level)
        return pnl[pnl <= var].mean() if len(pnl[pnl <= var]) > 0 else var

# --- 2. 主计算函数 ---
# (这个函数替代了原来的OptionVARSystem类，用于处理单次请求)

def calculate_var_for_option(ticker, expiry, strike, position_size, option_type):
    # 为确保每次结果一致，设置随机种子
    np.random.seed(42)
    
    results = {}
    try:
        # 获取数据
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # 动态获取期权链
        expirations = stock.options
        target_date = pd.Timestamp(expiry)
        closest_expiry = min(expirations, key=lambda d: abs(pd.Timestamp(d) - target_date))
        
        opt_chain = stock.option_chain(closest_expiry)
        chain = opt_chain.calls if option_type.lower() == 'call' else opt_chain.puts
        closest_option = chain.iloc[(chain['strike'] - strike).abs().argsort()[:1]]
        
        if closest_option.empty:
            raise ValueError(f"无法找到行权价接近 {strike} 的期权。")
        
        # --- 关键优化：使用Bid-Ask中间价 ---
        bid = closest_option['bid'].iloc[0]
        ask = closest_option['ask'].iloc[0]

        if bid == 0 or ask == 0:
             # 如果缺少bid/ask，则使用lastPrice作为备用，并给出提示
            option_price = closest_option['lastPrice'].iloc[0]
            price_source_info = f" (警告: 使用了可能滞后的lastPrice: ${option_price})"
        else:
            option_price = (bid + ask) / 2
            price_source_info = f" (基于Bid: ${bid}, Ask: ${ask})"

        actual_strike = closest_option['strike'].iloc[0]
        
        # 参数计算
        time_to_expiry = RiskCalculator.calculate_time_to_expiry(closest_expiry)
        risk_free_rate = TreasuryRateFetcher.get_risk_free_rate(closest_expiry)
        
        # --- 关键优化：严格的IV计算 + 备用逻辑 ---
        implied_volatility = ImpliedVolatilityCalculator.calculate_iv(option_price, current_price, actual_strike, time_to_expiry, risk_free_rate, option_type, chain=closest_option)
        if np.isnan(implied_volatility):
            raise ValueError(f"无法从市场价格反推出隐含波动率。市场价格可能不合理。价格来源: {price_source_info}")

        # 常量配置
        CONTRACT_MULTIPLIER = 100
        MC_SIMULATIONS = 10000
        TIME_HORIZON_DAYS = 1

        # 计算当前价值
        current_option_value = ImpliedVolatilityCalculator.black_scholes_price(current_price, strike, time_to_expiry, risk_free_rate, implied_volatility, option_type)
        total_position_value = current_option_value * position_size * CONTRACT_MULTIPLIER

        # 蒙特卡洛模拟
        simulated_prices = MonteCarloEngine.simulate_price_paths(current_price, risk_free_rate, implied_volatility, time_to_expiry, TIME_HORIZON_DAYS, MC_SIMULATIONS)
        simulated_option_values = MonteCarloEngine.calculate_option_values(simulated_prices, strike, time_to_expiry - (TIME_HORIZON_DAYS / 365.0), risk_free_rate, implied_volatility, option_type)
        simulated_position_values = simulated_option_values * position_size * CONTRACT_MULTIPLIER

        pnl = simulated_position_values - total_position_value
        
        # 计算风险指标
        var_95 = RiskCalculator.calculate_var(pnl, 0.95)
        cvar_95 = RiskCalculator.calculate_cvar(pnl, 0.95)
        var_99 = RiskCalculator.calculate_var(pnl, 0.99)
        cvar_99 = RiskCalculator.calculate_cvar(pnl, 0.99)
        
        # 计算Greeks
        greeks = GreeksCalculator.calculate_greeks(current_price, strike, time_to_expiry, risk_free_rate, implied_volatility, option_type)

        # 生成图表
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(pnl, bins=100, ax=ax, color='blue', alpha=0.7)
        ax.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: ${var_95:,.2f}')
        ax.axvline(var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: ${var_99:,.2f}')
        ax.set_title(f'{ticker.upper()} 持仓1日盈亏分布')
        ax.set_xlabel('Profit ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)

        # 填充结果字典
        results = {
            'success': True,
            'ticker': ticker.upper(),
            'expiry': closest_expiry,
            'strike': strike,
            'actual_strike': actual_strike,
            'position_size': position_size,
            'option_type': option_type.capitalize(),
            'current_price': round(current_price, 2),
            'current_option_price': round(option_price, 2),
            'price_source_info': price_source_info,
            'total_position_value': round(total_position_value, 2),
            'risk_free_rate': round(risk_free_rate * 100, 3),
            'implied_volatility': round(implied_volatility * 100, 2),
            'var_95': round(var_95, 2),
            'var_99': round(var_99, 2),
            'cvar_95': round(cvar_95, 2),
            'cvar_99': round(cvar_99, 2),
            'greeks': {k: round(v, 4) for k, v in greeks.items()},
            'chart_url': chart_url
        }

    except Exception as e:
        results = {'success': False, 'error': str(e)}

    return results

# --- 3. Flask Web应用部分 ---

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # 默认值
    defaults = {
        'ticker': 'MSTR',
        'expiry': (datetime.now() + pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
        'strike': '350',
        'position_size': '7',
        'option_type': 'call'
    }

    if request.method == 'POST':
        # 从表单获取用户输入
        ticker = request.form.get('ticker', 'MSTR')
        expiry = request.form.get('expiry')
        strike = float(request.form.get('strike', 350))
        position_size = int(request.form.get('position_size', 7))
        option_type = request.form.get('option_type', 'call')
        
        # 更新默认值以便在页面上显示用户最后输入
        defaults = request.form
        
        # 调用计算函数
        results = calculate_var_for_option(ticker, expiry, strike, position_size, option_type)
        
        return render_template('index.html', results=results, defaults=defaults)
    
    # 如果是GET请求，只显示页面
    return render_template('index.html', results=None, defaults=defaults)

if __name__ == '__main__':
    app.run(debug=True)
