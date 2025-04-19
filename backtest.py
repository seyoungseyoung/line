import numpy as np
import pandas as pd

def run_backtest(
    price_data: pd.DataFrame, # Changed from close_prices
    signals: pd.Series, 
    initial_capital=10000, 
    allow_fractional=True, 
    window=20, # Keep window param for potential S/R logic
    commission_rate=0.0001,
    stop_loss_pct: float = 0.05 # Added Stop-Loss percentage (e.g., 0.05 for 5%)
):
    """백테스트 실행 (수수료, 스탑로스 포함)""" # Updated docstring
    close_prices = price_data['Close']
    low_prices = price_data['Low'] # Need low prices for stop-loss check
    dates = close_prices.index
    T = len(close_prices)
    
    cash = initial_capital
    position = 0.0
    entry_price = None # Track entry price for stop-loss
    port_vals = [initial_capital]  
    trades = []
    stop_loss_trades_count = 0 # Counter for stop-loss trades
    
    # Use a fixed larger start index to ensure all signals are mature
    start_idx = 60 
    if start_idx >= T: 
        print(f"Warning: Data length ({T}) is shorter than start_idx ({start_idx}). No backtest possible.")
        return {
            'final_return': 0,
            'bh_return': 0,
            'excess_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0,
            'stop_loss_trades': 0, # Added
            'portfolio_values': np.full(T, initial_capital),
            'trades': []
        }

    port_vals = [initial_capital] * start_idx 
    
    # === Debug: Print signal range ===
    signals_slice = signals.iloc[start_idx:]
    print(f"Debug Backtest: signals_slice head:\n{signals_slice.head()}")
    try:
        print(f"Debug Backtest: signals_slice dtype: {signals_slice.dtype}") 
    except AttributeError:
        print(f"Debug Backtest: signals_slice dtype: {signals_slice.dtypes}") 
    # ============================
    # Calculate overall statistics for the slice
    sig_min = signals_slice.min() if not signals_slice.empty else pd.NA
    sig_max = signals_slice.max() if not signals_slice.empty else pd.NA
    sig_mean = signals_slice.mean() if not signals_slice.empty else pd.NA
    sig_std = signals_slice.std() if not signals_slice.empty and signals_slice.size > 1 else pd.NA

    print(f"Debug Backtest: Signal stats (post start_idx={start_idx}):\n" 
          f"Min: {f'{sig_min:.4f}' if pd.notna(sig_min) else 'NaN'}, "
          f"Max: {f'{sig_max:.4f}' if pd.notna(sig_max) else 'NaN'}, "
          f"Mean: {f'{sig_mean:.4f}' if pd.notna(sig_mean) else 'NaN'}, "
          f"Std: {f'{sig_std:.4f}' if pd.notna(sig_std) else 'NaN'}")
    # ==================================

    for i in range(start_idx, T):
        price = close_prices.iloc[i]
        current_low_price = low_prices.iloc[i] # Get current low
        prev_price = close_prices.iloc[i-1]
        signal = signals.iloc[i] 
        
        # Ensure current_low_price is scalar before use
        if isinstance(current_low_price, pd.Series):
            current_low_price = current_low_price.item() # Use .item() for single-element Series
        elif isinstance(current_low_price, pd.DataFrame):
            current_low_price = current_low_price.iloc[0,0] # Or handle DataFrame case if necessary
        
        # --- Stop-Loss Check --- 
        if position > 0 and entry_price is not None and stop_loss_pct > 0:
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            # Now comparison should be between two scalars
            if current_low_price <= stop_loss_price:
                # Stop-loss triggered: Sell position at stop_loss_price
                sell_price = stop_loss_price # Simulate execution at stop price
                trades.append((dates[i], sell_price, 'stop_sell'))
                traded_value = position * sell_price
                commission = traded_value * commission_rate
                cash += (traded_value - commission)
                position = 0
                entry_price = None # Reset entry price
                stop_loss_trades_count += 1
                
                # Update portfolio value after stop-loss and skip regular trading
                port_val = cash 
                port_vals.append(port_val)
                continue # Skip signal-based trading for this day
        # --- End Stop-Loss Check ---
        
        # 스칼라 값 추출 (필요 시)
        if isinstance(price, pd.Series): price = price.item() # Use .item() 
        if isinstance(prev_price, pd.Series): prev_price = prev_price.item()
        if isinstance(signal, pd.Series): signal = signal.item()
        
        portfolio_value_prev = cash + position * prev_price 
        prev_position = position
        prev_cash = cash
        traded_value = 0 
        trade_executed = False

        # --- Apply Trading Logic --- 
        if allow_fractional:
            target_position_value = portfolio_value_prev * signal 
            target_position = target_position_value / price if price > 0 else 0 
            diff = target_position - position
            
            if abs(diff * price) > 0.01: 
                if diff > 0: # Buy
                    if cash >= diff * price: 
                        trades.append((dates[i], price, 'buy'))
                        traded_value = diff * price
                        cash -= traded_value
                        # Update entry price only if starting a new position
                        if prev_position == 0: 
                            entry_price = price 
                        # How to handle avg price for fractional adds? Keep simple for now.
                        position += diff
                        trade_executed = True
                else: # Sell
                    trades.append((dates[i], price, 'sell'))
                    traded_value = abs(diff * price)
                    cash -= diff * price 
                    position += diff 
                    trade_executed = True
                    # Reset entry price if position is closed
                    if position == 0: 
                        entry_price = None
        else:
            # 이진 매매
            if signal > 0.6 and position == 0: 
                amount_to_buy = cash 
                shares_to_buy = amount_to_buy / price if price > 0 else 0
                potential_commission = (shares_to_buy * price) * commission_rate
                if shares_to_buy > 0 and cash >= (shares_to_buy * price) + potential_commission:
                    trades.append((dates[i], price, 'buy'))
                    traded_value = shares_to_buy * price
                    position = shares_to_buy
                    entry_price = price # Set entry price
                    commission = traded_value * commission_rate
                    cash -= (traded_value + commission) 
                    cash = max(0, cash) 
                    trade_executed = True
            elif signal < 0.4 and position > 0:
                trades.append((dates[i], price, 'sell'))
                traded_value = position * price
                commission = traded_value * commission_rate
                cash += (traded_value - commission) 
                position = 0
                entry_price = None # Reset entry price
                trade_executed = True
        
        # --- 수수료 차감 (for fractional only, binary includes it) ---
        if allow_fractional and trade_executed:
            commission = traded_value * commission_rate
            cash -= commission
        # --- -------- --- 
        
        # 포트폴리오 가치 업데이트
        port_val = cash + position * price
        port_vals.append(port_val)
    
    # 결과 계산
    port_vals_array = np.array(port_vals)
    port_vals_array[~np.isfinite(port_vals_array)] = 0 
    if len(port_vals_array) > 0 and port_vals_array[0] == 0 and initial_capital > 0:
        port_vals_array[0] = initial_capital 
        
    strat_returns = (port_vals_array / initial_capital - 1) * 100
    
    price_values = close_prices.values
    if len(price_values.shape) > 1:
        price_values = price_values.flatten()
        
    bh_final_return = 0
    if len(price_values) > start_idx:
        valid_bh_prices = price_values[start_idx:]
        if len(valid_bh_prices) > 0 and valid_bh_prices[0] != 0:
            bh_returns = (valid_bh_prices / valid_bh_prices[0] - 1) * 100
            bh_final_return = bh_returns[-1] if len(bh_returns) > 0 else 0

    final_return = strat_returns[-1] if len(strat_returns) > 0 else 0
    valid_port_vals = port_vals_array[start_idx:] 
    valid_port_vals = valid_port_vals[np.isfinite(valid_port_vals)]
    
    sharpe_ratio = 0
    max_drawdown = 0
    period_strat_return = 0
    if len(valid_port_vals) > 1:
        daily_returns = np.diff(valid_port_vals) / valid_port_vals[:-1]
        daily_returns[~np.isfinite(daily_returns)] = 0 
        daily_returns_std = np.std(daily_returns)
        if len(daily_returns) > 0 and daily_returns_std > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / daily_returns_std
        
        rolling_max = np.maximum.accumulate(valid_port_vals)
        rolling_max[rolling_max == 0] = 1 
        drawdowns = (valid_port_vals / rolling_max - 1) * 100
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        if valid_port_vals[0] != 0:
           period_strat_return = (valid_port_vals[-1] / valid_port_vals[0] - 1) * 100
           
    excess_return = period_strat_return - bh_final_return 
    
    return {
        'final_return': final_return, 
        'bh_return': bh_final_return, 
        'excess_return': excess_return, 
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades), # Total trades including stop-loss
        'stop_loss_trades': stop_loss_trades_count, # Added stop-loss count
        'portfolio_values': port_vals_array,
        'trades': trades
    } 