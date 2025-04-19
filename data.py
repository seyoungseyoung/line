import yfinance as yf
import pandas as pd
import warnings

# 불필요한 경고 무시
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def get_price_data(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """주가 데이터 다운로드 (종료일 포함)"""
    # Download data using yfinance, passing end_date if provided
    data = yf.download(
        ticker, 
        start=start_date, 
        end=end_date, # Pass end_date here
        progress=False, 
        auto_adjust=True,
        actions=False
    )
    # If end_date was used in download, no further filtering needed.
    # If end_date wasn't passed to yf.download but is provided,
    # you might filter here, but passing to yf.download is cleaner.
    # if end_date and not passed_to_yf_download: 
    #    data = data[:end_date]
    return data 