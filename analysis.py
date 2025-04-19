import pandas as pd
import numpy as np
from tqdm import tqdm

def find_support_resistance(prices: pd.DataFrame, window: int = 20, threshold: float = 0.005) -> pd.DataFrame:
    """지지선과 저항선 계산"""
    df = prices.copy()
    
    # 이동 최대/최소값 계산
    df['High_Max'] = df['High'].rolling(window=window).max()
    df['Low_Min'] = df['Low'].rolling(window=window).min()
    df['SMA'] = df['Close'].rolling(window=window).mean()
    
    # 볼린저 밴드 계산
    price_std = df['Close'].rolling(window=window).std()
    if isinstance(price_std, pd.DataFrame):
        price_std = price_std.iloc[:, 0]
    
    # NaN 값 처리 추가 (표준편차가 0일 경우 등)
    price_std = price_std.replace(0, np.nan).fillna(method='ffill').fillna(1e-6) # 0을 NaN으로 바꾸고 ffill, 그래도 남으면 작은 값
    
    upper_band = df['SMA'] + 2 * price_std
    lower_band = df['SMA'] - 2 * price_std
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    
    # 신호 초기화
    df['Support_Signal'] = 0.0
    df['Resistance_Signal'] = 0.0
    
    # 지지선 신호
    low_min_safe = df['Low_Min'].replace(0, np.nan).fillna(method='ffill').fillna(1e-6)
    low_ratio = df['Close'] / low_min_safe
    support_mask = (low_ratio >= 1.0) & (low_ratio <= 1.0 + threshold) & (~low_ratio.isna())
    df.loc[support_mask[support_mask].index, 'Support_Signal'] = 1.0
    
    # 저항선 신호
    high_max_safe = df['High_Max'].replace(0, np.nan).fillna(method='ffill').fillna(1e-6)
    high_ratio = df['Close'] / high_max_safe
    resistance_mask = (high_ratio >= 1.0 - threshold) & (high_ratio <= 1.0) & (~high_ratio.isna())
    df.loc[resistance_mask[resistance_mask].index, 'Resistance_Signal'] = 1.0
    
    # 볼린저 밴드 기반 신호 - NaN 방지 처리 강화
    band_low_ratio = (df['Close'] - df['Lower_Band']) / price_std 
    band_low_mask = (band_low_ratio <= 0.5) & (~band_low_ratio.isna())
    df.loc[band_low_mask[band_low_mask].index, 'Support_Signal'] += 0.5
    
    band_high_ratio = (df['Upper_Band'] - df['Close']) / price_std
    band_high_mask = (band_high_ratio <= 0.5) & (~band_high_ratio.isna())
    df.loc[band_high_mask[band_high_mask].index, 'Resistance_Signal'] += 0.5
    
    # 신호 정규화
    df['Support_Signal'] = df['Support_Signal'].clip(0, 1)
    df['Resistance_Signal'] = df['Resistance_Signal'].clip(0, 1)
    
    # Return only necessary columns used elsewhere, ensure basic NaN fill
    result_cols = ['SMA', 'Upper_Band', 'Lower_Band', 'Support_Signal', 'Resistance_Signal']
    final_df = df[result_cols].copy()
    # Fill any remaining NaNs (e.g., at the very beginning)
    final_df = final_df.fillna(method='bfill').fillna(0) # Backfill first, then fill rest with 0
    return final_df

def calculate_regression_channel(price_data: pd.DataFrame, window: int = 20):
    """선형 회귀 채널 및 신호 계산."""
    df = price_data.copy()
    # 회귀 분석은 로그 가격으로 수행하는 것이 일반적
    log_close = np.log(df['Close'].replace(0, 1e-9)) # 0 값 방지
    n = len(df)
    
    # 결과 컬럼 초기화
    reg_center = np.full(n, np.nan)
    reg_upper = np.full(n, np.nan)
    reg_lower = np.full(n, np.nan)
    reg_signal = np.full(n, 0.5) # 기본값 0.5

    # 독립 변수 (시간: 0, 1, ..., window-1)
    x = np.arange(window)
    A = np.vstack([x, np.ones(len(x))]).T
    
    # tqdm 추가하여 진행 상황 표시
    for i in tqdm(range(window - 1, n), desc=f"Reg Channel (W:{window})", leave=False, mininterval=1.0):
        y_window = log_close.iloc[i - window + 1 : i + 1].values
        
        # NaN 값이나 Inf 값이 있는지 확인
        if np.any(~np.isfinite(y_window)):
            continue # 유효하지 않은 데이터가 있으면 건너뛰기
            
        try:
            # 선형 회귀: y = mx + c
            m, c = np.linalg.lstsq(A, y_window, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue # 회귀 실패 시 건너뛰기

        # 현재 시점(window의 마지막 지점)에서의 예측값 계산
        predicted_log_price = m * (window - 1) + c 
        
        # 추정 표준 오차 계산
        residuals = y_window - (m * x + c)
        # 분모가 0 또는 음수가 되는 경우 방지 (데이터가 완벽히 선형일 때)
        if window <= 2:
             std_err = 0 # 샘플 수가 너무 작으면 표준 오차 0으로 처리
        else:
            residual_sum_sq = np.sum(residuals**2)
            if residual_sum_sq < 1e-9: # 거의 완벽한 선형일 때
                 std_err = 1e-6 # 매우 작은 값으로 설정
            else:
                 std_err = np.sqrt(residual_sum_sq / (window - 2))
                 
        # 채널 경계 계산 (예: 중심선 +/- 1 표준 오차)
        upper_log = predicted_log_price + std_err
        lower_log = predicted_log_price - std_err
        
        # 결과를 원래 가격 스케일로 저장
        reg_center[i] = np.exp(predicted_log_price)
        reg_upper[i] = np.exp(upper_log)
        reg_lower[i] = np.exp(lower_log)
        
        # 신호 계산: 채널 내 현재 가격 위치 (0~1)
        channel_height = upper_log - lower_log
        if channel_height > 1e-9: # 채널 높이가 0에 가까운 경우 방지
            current_log_price = log_close.iloc[i]
            signal = (current_log_price - lower_log) / channel_height
            reg_signal[i] = np.clip(signal, 0, 1)
        # 높이가 0에 가까우면 기본값 0.5 유지

    # DataFrame으로 변환
    channel_df = pd.DataFrame({
        'Reg_Center': reg_center,
        'Reg_Upper': reg_upper,
        'Reg_Lower': reg_lower,
        'Reg_Signal': reg_signal
    }, index=df.index)
    
    # 초기 NaN 값 처리 강화 (신호는 0.5로, 선은 bfill 후 ffill)
    channel_df['Reg_Signal'] = channel_df['Reg_Signal'].fillna(0.5)
    channel_df = channel_df.fillna(method='bfill').fillna(method='ffill').fillna(0) # Backfill, Forward fill, then 0

    return channel_df

def calculate_macd_signal(
    close_prices: pd.Series, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9,
    normalization_window: int = 50
) -> pd.Series:
    """MACD 기반 추세 신호 계산 (정규화된 히스토그램 사용)"""
    # EMA 계산
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD 라인 및 시그널 라인
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # MACD 히스토그램
    histogram = macd_line - signal_line
    
    # 히스토그램 정규화 (롤링 윈도우 사용)
    rolling_min = histogram.rolling(window=normalization_window, min_periods=1).min()
    rolling_max = histogram.rolling(window=normalization_window, min_periods=1).max()
    
    # 분모가 0이 되는 경우 방지
    denominator = rolling_max - rolling_min
    denominator[denominator == 0] = 1  # 분모가 0이면 정규화 결과는 0이 됨 (histogram - rolling_min = 0)
    
    normalized_signal = (histogram - rolling_min) / denominator
    
    # 초기 NaN 값 처리 (예: 중간값 0.5로 채우기)
    normalized_signal = normalized_signal.fillna(0.5)
    
    # 0~1 범위로 클리핑
    normalized_signal = normalized_signal.clip(0, 1)
    
    return normalized_signal 