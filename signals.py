import pandas as pd
import numpy as np

def calculate_combined_signal(
    trend_signal: pd.Series | None,
    support_signal: pd.Series | None,
    resistance_signal: pd.Series | None,
    regression_signal: pd.Series | None, 
    weights: tuple
) -> pd.Series:
    """가중치 기반으로 최대 4개 신호 결합"""
    # Ensure weights tuple has 4 elements, padding with 0 if necessary
    weights = tuple(list(weights) + [0] * (4 - len(weights)))
    trend_weight, support_weight, resistance_weight, regression_weight = weights
    
    signals_to_combine = []
    effective_weights = []
    indices = [] # To store indices of signals being used

    # Align signals and collect non-zero weighted signals
    base_index = None
    temp_signals = {}
    if trend_signal is not None and trend_weight != 0:
        trend_series = trend_signal.iloc[:, 0] if isinstance(trend_signal, pd.DataFrame) else trend_signal
        temp_signals['trend'] = trend_series
        base_index = trend_series.index
    if support_signal is not None and support_weight != 0:
        temp_signals['support'] = support_signal
        if base_index is None: base_index = support_signal.index
    if resistance_signal is not None and resistance_weight != 0:
        temp_signals['resistance'] = resistance_signal
        if base_index is None: base_index = resistance_signal.index
    if regression_signal is not None and regression_weight != 0:
        # Use (1 - Reg_Signal) so lower channel (0) contributes positively like support
        temp_signals['regression'] = 1 - regression_signal 
        if base_index is None: base_index = regression_signal.index
        
    # Ensure a base index exists (at least one signal must be provided)
    if base_index is None:
        # Or raise an error, or return a default series
        print("Warning: No valid signals provided for combination.")
        # Attempt to find an index from any input signal even if weight is 0
        all_signals = [trend_signal, support_signal, resistance_signal, regression_signal]
        for sig in all_signals:
            if sig is not None:
                 base_index = sig.index
                 break
        if base_index is None:
             raise ValueError("Cannot determine index for combined signal.")
        return pd.Series(0.5, index=base_index) # Return neutral signal

    # Align all signals to the base index and prepare for combination
    aligned_signals = {}
    for name, sig in temp_signals.items():
        aligned_signals[name] = sig.reindex(base_index).fillna(0.5) # fill NaN with neutral 0.5

    # Prepare lists for weighted sum
    combined = pd.Series(0.0, index=base_index)
    total_weight = 0.0

    if 'trend' in aligned_signals:
        combined += trend_weight * aligned_signals['trend']
        total_weight += abs(trend_weight)
    if 'support' in aligned_signals:
        combined += support_weight * aligned_signals['support']
        total_weight += abs(support_weight)
    if 'regression' in aligned_signals: # Note: using (1-reg) here
        combined += regression_weight * aligned_signals['regression'] 
        total_weight += abs(regression_weight)
    if 'resistance' in aligned_signals:
        # Resistance acts negatively
        combined -= resistance_weight * aligned_signals['resistance'] 
        total_weight += abs(resistance_weight)

    # Normalize by total absolute weight if total_weight is not zero
    if total_weight > 1e-9:
        # Scale combined signal - this step might need refinement based on desired output range
        # Simple scaling might not be ideal. Let's just combine for now and clip.
        # combined = combined / total_weight # Normalization - Revisit this logic
        pass # Skip normalization for now, rely on clipping
    else:
        # If total weight is zero (or very close), return neutral signal
        return pd.Series(0.5, index=base_index)

    # Clip the result to 0-1 range and fill any remaining NaNs
    return combined.clip(0, 1).fillna(0.5) 