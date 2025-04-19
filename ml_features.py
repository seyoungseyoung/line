import pandas as pd
import numpy as np
# Import existing analysis functions if they return relevant data
from analysis import find_support_resistance, calculate_macd_signal, calculate_regression_channel

def safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Performs division between two Series, handling zero denominators."""
    # Ensure denominator is safe
    den_safe = denominator.replace(0, np.nan)
    # Perform division (Pandas aligns by index automatically)
    result = numerator / den_safe
    return result.replace([np.inf, -np.inf], np.nan)

def create_features(price_data: pd.DataFrame, sr_window: int = 10, macd_params: dict = None, reg_window: int = 20):
    """
    Calculates features for the ML model from price data.
    Focus on robust calculation and proper DataFrame construction.
    """
    print(f"Creating features (SR Win: {sr_window}, Reg Win: {reg_window})...")
    df = price_data.copy()
    close = df['Close'].squeeze() # Ensure close is Series
    features_df = pd.DataFrame(index=df.index) 
    
    # --- Reuse existing indicators ---
    # 1. S/R Signals
    try:
        sr_data = find_support_resistance(df, window=sr_window)
        # Ensure columns from sr_data are Series before using
        support_sig = sr_data['Support_Signal'].squeeze()
        resistance_sig = sr_data['Resistance_Signal'].squeeze()
        sma = sr_data['SMA'].squeeze()
        upper_band = sr_data['Upper_Band'].squeeze()
        lower_band = sr_data['Lower_Band'].squeeze()
        
        features_df['Support_Signal'] = support_sig
        features_df['Resistance_Signal'] = resistance_sig
        features_df['Price_vs_SMA'] = safe_division(close, sma) - 1
        features_df['Price_vs_UpperBand'] = safe_division(close, upper_band) - 1
        features_df['Price_vs_LowerBand'] = safe_division(close, lower_band) - 1
    except Exception as e:
        print(f"Error calculating S/R features: {e}. Setting to NaN.")
        for col in ['Support_Signal', 'Resistance_Signal', 'Price_vs_SMA', 'Price_vs_UpperBand', 'Price_vs_LowerBand']:
            features_df[col] = np.nan 

    # 2. MACD Signal
    try:
        macd_params = macd_params or {}
        features_df['MACD_Signal'] = calculate_macd_signal(close, **macd_params).squeeze()
    except Exception as e:
        print(f"Error calculating MACD features: {e}. Setting to NaN.")
        features_df['MACD_Signal'] = np.nan

    # 3. Regression Channel Signal
    try:
        channel_data = calculate_regression_channel(df, window=reg_window)
        # Ensure columns from channel_data are Series before using
        reg_sig = channel_data['Reg_Signal'].squeeze()
        reg_center = channel_data['Reg_Center'].squeeze()
        reg_upper = channel_data['Reg_Upper'].squeeze()
        reg_lower = channel_data['Reg_Lower'].squeeze()
        
        features_df['Reg_Signal'] = reg_sig
        features_df['Price_vs_RegCenter'] = safe_division(close, reg_center) - 1
        features_df['Price_vs_RegUpper'] = safe_division(close, reg_upper) - 1
        features_df['Price_vs_RegLower'] = safe_division(close, reg_lower) - 1
    except Exception as e:
        print(f"Error calculating Regression features: {e}. Setting to NaN.")
        for col in ['Reg_Signal', 'Price_vs_RegCenter', 'Price_vs_RegUpper', 'Price_vs_RegLower']:
            features_df[col] = np.nan

    # --- Add New Features ---
    # 4. Lagged Returns
    for lag in [1, 3, 5, 10, 21]: 
        try:
            features_df[f'Return_{lag}d'] = close.pct_change(lag)
        except Exception as e:
            print(f"Error calculating Return_{lag}d: {e}. Setting to NaN.")
            features_df[f'Return_{lag}d'] = np.nan
        
    # 5. Volatility
    try:
        # Ensure close.shift(1) is a Series for safe_division
        shifted_close = close.shift(1)
        log_returns = np.log(safe_division(close, shifted_close))
        features_df['Volatility_21d'] = log_returns.rolling(window=21).std() * np.sqrt(252)
    except Exception as e:
        print(f"Error calculating Volatility features: {e}. Setting to NaN.")
        features_df['Volatility_21d'] = np.nan

    # 6. Price relative to moving averages
    for ma_window in [10, 20, 50]:
        try:
            sma_series = close.rolling(window=ma_window).mean().squeeze()
            features_df[f'Price_vs_SMA{ma_window}'] = safe_division(close, sma_series) - 1
        except Exception as e:
            print(f"Error calculating Price_vs_SMA{ma_window} features: {e}. Setting to NaN.")
            features_df[f'Price_vs_SMA{ma_window}'] = np.nan
            
    # --- Clean up and return features ---
    # Fill NaNs from calculation steps
    features_df = features_df.fillna(method='ffill').fillna(method='bfill')
    
    all_nan_cols = features_df.columns[features_df.isna().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns {all_nan_cols} are entirely NaN after fill. Filling with 0.")
        features_df[all_nan_cols] = features_df[all_nan_cols].fillna(0)
        
    # IMPORTANT: Shift features by 1 day for prediction
    features_final_shifted = features_df.shift(1)
    
    print(f"Generated {features_final_shifted.shape[1]} features (before final dropna).")
    # Return the shifted features, let the calling function handle final dropna with target
    return features_final_shifted


def create_target(close_prices: pd.Series, future_days: int = 5):
    """
    Creates the target variable: 1 if price increases after future_days, 0 otherwise.
    Uses the 'Close' price for calculation.
    Returns a Pandas Series named correctly.
    """
    print(f"Creating target variable ({future_days} days ahead)...")
    # Calculate future return: Price[t+k] / Price[t]
    # Ensure close_prices is treated as a Series
    close_series = pd.Series(close_prices)
    future_return = close_series.shift(-future_days) / close_series
    
    # Target = 1 if future return > 1 (price increased), 0 otherwise
    target_values = (future_return > 1.0).astype(int)
    
    # Create a Series with the correct name and index
    target_series = pd.Series(target_values, index=close_series.index, name=f'Target_{future_days}d_Up')
    
    return target_series

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    # This part is only executed when running ml_features.py directly
    from data import get_price_data
    ticker = 'SPY'
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    price_data = get_price_data(ticker, start_date, end_date)
    
    # Create features using specific windows
    features = create_features(price_data, sr_window=10, reg_window=20)
    
    # Create target variable (predict if price goes up in 5 days)
    target = create_target(price_data['Close'], future_days=5)
    
    # Combine features and target
    # Need to align indices as target calculation shifts data backward
    combined_data = pd.concat([features, target], axis=1)
    
    # Drop rows with NaN values created by feature/target calculation
    # This happens at the beginning (due to rolling features/lags/shifts) 
    # and at the end (due to future target lookup)
    original_len = len(combined_data)
    combined_data = combined_data.dropna()
    print(f"\nOriginal data length: {original_len}")
    print(f"Data length after NaN drop: {len(combined_data)}")
    
    print("\nCombined Data Head:")
    print(combined_data.head())
    
    print("\nCombined Data Tail:")
    print(combined_data.tail())
    
    print("\nFeature Correlations with Target:")
    # Ensure target column exists before calculating correlation
    if target.name in combined_data.columns:
         correlations = combined_data.corr()[target.name].sort_values(ascending=False)
         print(correlations)
    else:
         print("Target column not found for correlation calculation.") 