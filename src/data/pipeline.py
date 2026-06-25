import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List

class DataPipeline:
    def __init__(self, start_date: str = "2014-01-01", end_date: str = "2024-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        # 29 DJIA Constituents excluding NVDA
        self.tickers: List[str] = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "UNH", "XOM", "JPM", "V", "PG",
            "AVGO", "TSLA", "JNJ", "COST", "HD", "MRK", "NFLX", "AMD", "PEP", "ADBE",
            "TMO", "CVX", "WMT", "BAC", "MCD", "CRM", "ABT", "DIS", "CSCO"
        ]
        
    def fetch_raw_data(self) -> pd.DataFrame:
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=False)
        return data

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unique_dates = df.index.unique()
        n_total = len(unique_dates)
        n_train = int(n_total * 0.60) # 1640 days
        n_val = int(n_total * 0.20)   # 547 days
        
        train_dates = unique_dates[:n_train]
        val_dates = unique_dates[n_train:n_train + n_val]
        test_dates = unique_dates[n_train + n_val:]
        
        return df.loc[train_dates], df.loc[val_dates], df.loc[test_dates]