import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import itertools

from data import get_price_data
from analysis import find_support_resistance, calculate_macd_signal, calculate_regression_channel
from signals import calculate_combined_signal
from backtest import run_backtest
from visualization import plot_top_strategies

def optimize_strategy(
    ticker, 
    start_date, 
    end_date=None,
    windows=[10, 20, 30], 
    plot_results=True,
    stop_loss_pct=0.05
):
    """다양한 전략 조합을 테스트하고 최적화 (회귀 채널 포함)"""
    print(f"Downloading: {ticker} data ({start_date} ~ Present)...")
    price_data = get_price_data(ticker, start_date, end_date)
    
    close_prices = price_data['Close']
    
    # 모든 지지/저항선 데이터 미리 계산 및 시프트
    sr_data_by_window = {}
    shifted_support_signals = {}
    shifted_resistance_signals = {}
    for window in tqdm(windows, desc="S/R Calc & Shift"):
        sr_data = find_support_resistance(price_data, window=window)
        sr_data_by_window[window] = sr_data # Store original for potential other uses
        # Shift signals by 1 day to avoid lookahead bias, fill initial NaN with 0.5
        shifted_support_signals[window] = sr_data['Support_Signal'].shift(1).fillna(0.5) 
        shifted_resistance_signals[window] = sr_data['Resistance_Signal'].shift(1).fillna(0.5)
    
    # Calculate MACD signal once and shift
    print("Calculating MACD signal...")
    macd_signal_original = calculate_macd_signal(close_prices) # Using default MACD params
    # Shift signal by 1 day, fill initial NaN with 0.5
    macd_signal_shifted = macd_signal_original.shift(1).fillna(0.5) 
    
    # Calculate Regression Channel signals
    print("Calculating Regression Channel signals...")
    shifted_regression_signals = {}
    for window in tqdm(windows, desc="RegChannel Calc & Shift"):
        channel_data = calculate_regression_channel(price_data, window=window)
        shifted_regression_signals[window] = channel_data['Reg_Signal'].shift(1).fillna(0.5)
    
    # 테스트할 전략 조합 (4-element weights: trend, support, resistance, regression)
    print("Generating strategy combinations...")
    strategies = []
    # Define base signals and combinations using 4-element weights (M, S, R, G)
    base_weights = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)] # Single signals
    combo_weights = [
        (0.5,0.5,0,0), (0.5,0,0.5,0), (0.5,0,0,0.5), # MACD + others
        (0,0.5,0.5,0), (0,0.5,0,0.5), (0,0,0.5,0.5), # Sup/Res + Reg
        (0.4,0.3,0,0.3), (0.4,0,0.3,0.3), (0.3,0.3,0.1,0.3), # 3-signal combos
        (0.25, 0.25, 0.25, 0.25) # All 4 signals
    ]

    # Helper to create names based on non-zero weights
    def get_strategy_name(w, win):
        parts = []
        labels = ["MACD", "Sup", "Res", "Reg"]
        for i, weight in enumerate(w):
            # Consider weight significant if abs > epsilon? For now, > 0
            if abs(weight) > 1e-9:
                 parts.append(labels[i])
        name = "+".join(parts) if parts else "Neutral" # Name for all zero weights
        return f"{name} (W:{win})"

    for window in windows:
        # Add single signal strategies
        for w in base_weights:
             strategies.append({'name': get_strategy_name(w, window), 'weights': w, 'window': window})
        # Add combination strategies
        for w in combo_weights:
             strategies.append({'name': get_strategy_name(w, window), 'weights': w, 'window': window})

    # Remove potential duplicates (if different combos lead to same effective strategy/name)
    unique_strategies_set = {tuple(sorted(d.items())) for d in strategies}
    strategies = [dict(t) for t in unique_strategies_set]
    # Sort for consistent order (optional)
    strategies.sort(key=lambda x: (x['window'], x['name']))

    # 전략 평가 결과 저장
    results = []
    
    # 각 전략 테스트
    print(f"\nRunning backtests for {len(strategies)} strategies (Stop-Loss: {stop_loss_pct*100}%)...")
    for strategy in tqdm(strategies, desc="Strategy Backtests"):
        start_time = time.time()
        window = strategy['window']
        weights = strategy['weights'] # Should be 4 elements now
        
        # Get shifted signals for the current window
        trend_sig = macd_signal_shifted # MACD is window-independent calculation wise
        support_sig = shifted_support_signals.get(window)
        resistance_sig = shifted_resistance_signals.get(window)
        regression_sig = shifted_regression_signals.get(window)
        
        # Skip if essential signals for the strategy are missing
        # (MACD signal is assumed to always exist)
        if (weights[1] != 0 and support_sig is None) or \
           (weights[2] != 0 and resistance_sig is None) or \
           (weights[3] != 0 and regression_sig is None):
            # print(f"Skipping strategy {strategy['name']} due to missing signal data for window {window}") # Optional debug
            continue
        
        # Combine signals (Pass all 4, function handles None/zero weights)
        combined_signal = calculate_combined_signal(
            trend_sig, support_sig, resistance_sig, regression_sig, weights
        )
        
        # 백테스트 실행
        result = run_backtest(
            price_data, 
            combined_signal, 
            window=window,
            stop_loss_pct=stop_loss_pct
        )
        
        # 결과 저장 (Store original signals if needed for analysis later, TBD)
        result.update({
            'strategy_name': strategy['name'],
            'weights': weights, # Store the 4-element weights
            'window': window,
            'computation_time': time.time() - start_time
        })
        
        # Add a check for valid result structure before appending
        if isinstance(result, dict) and 'excess_return' in result:
             results.append(result)
        else:
             print(f"Warning: Backtest for {strategy['name']} returned invalid result: {result}")
    
    # 결과 정렬 및 출력 (초과 수익률 기준)
    sorted_results = sorted(results, key=lambda x: x.get('excess_return', -np.inf), reverse=True)
    
    print("\n===== Top 5 Strategies =====")
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. {result['strategy_name']}")
        # Safely format weights, handle potential missing keys or wrong length
        try:
             weights_tuple = result.get('weights', (0,0,0,0))
             # Ensure it's a tuple/list of expected length before formatting
             if isinstance(weights_tuple, (list, tuple)) and len(weights_tuple) == 4:
                  weights_str = f"({weights_tuple[0]:.2f}, {weights_tuple[1]:.2f}, {weights_tuple[2]:.2f}, {weights_tuple[3]:.2f})"
             else:
                  weights_str = str(weights_tuple) # Fallback to string representation
        except Exception:
             weights_str = "N/A"
        print(f"   Weights (M, S, R, G): {weights_str}") # Updated labels
        print(f"   Final Return: {result.get('final_return', 0):.2f}%, B&H: {result.get('bh_return', 0):.2f}%, " +
              f"Excess Return: {result.get('excess_return', 0):.2f}%")
        print(f"   Sharpe: {result.get('sharpe_ratio', 0):.3f}, Max Drawdown: {result.get('max_drawdown', 0):.2f}%, " +
              f"Trades: {result.get('num_trades', 0)}, Stops: {result.get('stop_loss_trades', 0)}") # Added stops
        print()
    
    # 엘보우 곡선 분석 (추세선 개수와 성능 관계)
    if plot_results and sorted_results: # Check if results exist
        try:
            # 2) 상위 5개 전략 성능 비교
            plot_top_strategies(sorted_results[:5], close_prices, ticker)
        except Exception as e:
            print(f"Plotting error: {e}")
    
    # 최고 전략 반환 (결과가 있을 경우)
    return sorted_results[0] if sorted_results else None 