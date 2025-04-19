import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_top_strategies(top_results, close_prices, ticker):
    """상위 전략들의 포트폴리오 가치 비교 시각화"""
    plt.figure(figsize=(12, 8))
    
    # 기준 가격 (초기값 = 10000)
    initial_value = 10000
    price_series = pd.Series(close_prices) if isinstance(close_prices, np.ndarray) else close_prices
    price_scaled = price_series / price_series.iloc[0] * initial_value
    
    # 매수 후 보유 전략
    plt.plot(price_series.index, price_scaled, 
             color='black', linestyle='--', label='Buy & Hold')
    
    # 상위 전략들
    for i, result in enumerate(top_results):
        portfolio_values = result['portfolio_values']
        strategy_name = result['strategy_name']
        
        plt.plot(price_series.index, portfolio_values, 
                 label=f"{i+1}. {strategy_name} ({result['excess_return']:.1f}% Excess)")
    
    plt.title(f'{ticker} Top Strategies Portfolio Value Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_best_strategy(price_data, close_prices, 
                          trend_signal, support_signal, resistance_signal, 
                          combined_signal, result, strategy_desc, weights, 
                          window, ticker):
    """최적 전략 상세 시각화"""
    trend_weight, support_weight, resistance_weight = weights

    try:
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True, 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 가격 차트 및 지지/저항선
        ax1 = axs[0]
        ax1.plot(close_prices.index, close_prices, color='black', linewidth=1, label='Price')
        
        if support_weight > 0 or resistance_weight > 0:
            ax1.plot(price_data.index, price_data['SMA'], color='blue', linewidth=1, alpha=0.7, label='SMA')
            ax1.plot(price_data.index, price_data['Upper_Band'], color='red', linewidth=1, alpha=0.5, label='Upper Band')
            ax1.plot(price_data.index, price_data['Lower_Band'], color='green', linewidth=1, alpha=0.5, label='Lower Band')
        
        # 거래 표시 (최대 50개)
        trades = result['trades']
        max_markers = 50
        if len(trades) > max_markers:
            sample_step = len(trades) // max_markers
            sampled_trades = trades[::sample_step]
        else:
            sampled_trades = trades
            
        for dt, p, action in sampled_trades:
            if action == 'buy':
                ax1.scatter(dt, p, marker='^', color='green', s=80, alpha=0.7)
            else:
                ax1.scatter(dt, p, marker='v', color='red', s=80, alpha=0.7)
                
        ax1.set_ylabel('Price')
        ax1.set_title(f"{ticker}: Analysis of Best Strategy '{strategy_desc}'")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 신호 차트
        ax2 = axs[1]
        if trend_weight > 0:
            ax2.plot(trend_signal.index, trend_signal, label='MACD', color='purple', alpha=0.7)
        if support_weight > 0:
            ax2.plot(support_signal.index, support_signal, label='Support', color='green', alpha=0.7)
        if resistance_weight > 0:
            ax2.plot(resistance_signal.index, resistance_signal, label='Resistance', color='red', alpha=0.7)
        
        ax2.plot(combined_signal.index, combined_signal, label='Combined', color='blue', linewidth=2)
        ax2.set_ylabel('Signal Strength')
        ax2.set_title("Trading Signals")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        # 포트폴리오 가치
        ax3 = axs[2]
        
        # 포트폴리오 가치와 매수 후 보유 비교
        portfolio_values = result['portfolio_values']
        price_scaled = close_prices / close_prices.iloc[0] * 10000
        
        ax3.plot(close_prices.index, portfolio_values, label='Strategy', linewidth=1.5)
        ax3.plot(close_prices.index, price_scaled, label='Buy & Hold', linestyle='--', linewidth=1.5)
        
        ax3.set_ylabel('Portfolio Value')
        ax3.set_xlabel('Date')
        ax3.set_title(f"Portfolio Value (Initial: $10,000)")
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close('all')  # 메모리 정리
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}") 