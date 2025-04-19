import pandas as pd
import numpy as np
from datetime import datetime

from data import get_price_data
from analysis import find_support_resistance, calculate_macd_signal, calculate_regression_channel
from analysis import find_support_resistance, calculate_macd_signal
from signals import calculate_combined_signal
from backtest import run_backtest
from visualization import visualize_best_strategy

def get_strategy_description(weights, window):
    """Generate strategy description string (handles 4 weights)"""
    # Ensure weights has 4 elements
    weights = tuple(list(weights) + [0] * (4 - len(weights)))
    trend_weight, support_weight, resistance_weight, regression_weight = weights

    parts = []
    labels = ["MACD", "Support", "Resistance", "Regression"]
    # Check absolute weight to handle potential negative weights if logic changes later
    if abs(trend_weight) > 1e-9: parts.append(labels[0])
    if abs(support_weight) > 1e-9: parts.append(labels[1])
    if abs(resistance_weight) > 1e-9: parts.append(labels[2])
    if abs(regression_weight) > 1e-9: parts.append(labels[3])

    strategy_desc = "+".join(parts) if parts else "Neutral" # Use Neutral if all weights are zero
    return f"{strategy_desc} (W:{window})"

def analyze_best_strategy(
    ticker, 
    start_date, 
    end_date=None, 
    window=20, 
    weights=(0.4, 0.3, 0.3, 0.0),
    stop_loss_pct=0.05
):
    """Detailed analysis and visualization of the best strategy (incl. Regression)"""
    # Ensure weights tuple has 4 elements
    weights = tuple(list(weights) + [0] * (4 - len(weights)))
    trend_weight, support_weight, resistance_weight, regression_weight = weights

    print(f"Downloading: {ticker} data...")
    price_data = get_price_data(ticker, start_date, end_date)
    close_prices = price_data['Close']
    
    # Calculate support/resistance
    print(f"Calculating Support/Resistance (window={window})...")
    sr_data = find_support_resistance(price_data, window=window)
    
    # Calculate MACD signal
    print(f"Calculating MACD signal...")
    trend_signal_original = calculate_macd_signal(close_prices)
    
    # Calculate Regression Channel
    print(f"Calculating Regression Channel (window={window})...")
    channel_data = calculate_regression_channel(price_data, window=window)
    
    # === Shift signals (Prevent lookahead bias) ===
    trend_signal_shifted = trend_signal_original.shift(1).fillna(0.5)
    support_signal_shifted = sr_data['Support_Signal'].shift(1).fillna(0.5)
    resistance_signal_shifted = sr_data['Resistance_Signal'].shift(1).fillna(0.5)
    # Use the signal column from channel_data
    regression_signal_original = channel_data['Reg_Signal'] # Keep original for viz
    regression_signal_shifted = regression_signal_original.shift(1).fillna(0.5)
    # =========================================

    # Combined signal (Use ALL shifted signals)
    combined_signal_shifted = calculate_combined_signal(
        trend_signal_shifted,
        support_signal_shifted,
        resistance_signal_shifted,
        regression_signal_shifted, # Add shifted regression signal
        weights
    )
    
    # Run backtest (Use shifted combined signal)
    print(f"Running backtest (Stop-Loss: {stop_loss_pct*100}%)...")
    result = run_backtest(
        price_data,
        combined_signal_shifted,
        window=window,
        stop_loss_pct=stop_loss_pct
    )
    
    # Generate strategy description
    strategy_desc = get_strategy_description(weights, window)
    
    # Visualize results (Pass necessary data including channel_data)
    print("Visualizing best strategy details...")
    # Check if the required visualization function expects these args
    visualize_best_strategy(
        price_data, # Pass original price data (contains Close, Low etc.)
        sr_data, # Contains SMA, Bands, S/R signals
        channel_data, # Contains Reg_Center, Upper, Lower, Reg_Signal
        trend_signal_original,  # Original MACD signal
        # Pass original individual signals if needed by visualization
        # support_signal_original = sr_data['Support_Signal'],
        # resistance_signal_original = sr_data['Resistance_Signal'],
        regression_signal_original, # Original Regression signal
        combined_signal_shifted, # Shifted combined signal used for trades
        result,
        strategy_desc,
        weights, # Pass 4-element weights
        window,
        ticker
    )
    
    # Performance Summary
    print("\n===== Strategy Performance Summary =====")
    print(f"Strategy: {strategy_desc}")
    weights_str = f"({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}, {weights[3]:.2f})"
    print(f"Weights (M, S, R, G): {weights_str}")
    print(f"Final Return: {result['final_return']:.2f}%")
    print(f"Buy & Hold Return: {result['bh_return']:.2f}%")
    print(f"Excess Return: {result['excess_return']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
    print(f"Number of Trades: {result['num_trades']}")
    print(f"Stop-Loss Trades: {result['stop_loss_trades']}")
    
    # Add explanation for the best strategy signals
    if support_weight > 0:
        print("\n===== Signal Explanation (Support) =====")
        print(f"1. Rolling Low: Lowest low over past {window} days.")
        print("2. Proximity Check: Signal(1.0) if Close is 0% to 0.5% above Rolling Low.")
        print("3. Bollinger Band: Adds +0.5 if Close is near/below Lower Band.")
    if resistance_weight > 0:
        print("\n===== Signal Explanation (Resistance) =====")
        print(f"1. Rolling High: Highest high over past {window} days.")
        print("2. Proximity Check: Signal(1.0) if Close is 0% to 0.5% below Rolling High.")
        print("3. Bollinger Band: Adds +0.5 if Close is near/above Upper Band.")
    if regression_weight > 0:
        print("\n===== Signal Explanation (Regression Channel) =====")
        print(f"1. Linear Regression: Calculated on log prices over past {window} days.")
        print("2. Channel: Center line +/- 1 std error.")
        print("3. Signal (Reg_Signal): Position within channel (0=Lower, 0.5=Center, 1=Upper).")
        print("4. Contribution: Weight applied to (1 - Reg_Signal); stronger near lower band.")
    
    return result 