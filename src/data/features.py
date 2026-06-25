import numpy as np
import pandas as pd
from numba import njit

class FeatureEngineer:
    def __init__(self, window_size: int = 15):
        self.w = window_size

    def compute_technical_indicators(self, df: pd.DataFrame) -> np.ndarray:
        # Expected structure input: MultiIndex columns (Metric, Ticker)
        tickers = df['Adj Close'].columns
        n_assets = len(tickers)
        n_days = len(df)
        n_features = 12
        
        # Output State Tensor: (n_days, n_assets, n_features)
        state_matrix = np.zeros((n_days, n_assets, n_features))
        
        for idx, ticker in enumerate(tickers):
            adj_close = df['Adj Close'][ticker].values
            volume = df['Volume'][ticker].values
            
            # Returns
            ret = np.zeros_like(adj_close)
            ret[1:] = np.log(adj_close[1:] / adj_close[:-1])
            
            # EMAs (10, 20, 50, 100, 200)
            ema10 = df['Adj Close'][ticker].ewm(span=10, adjust=False).mean().values
            ema20 = df['Adj Close'][ticker].ewm(span=20, adjust=False).mean().values
            ema50 = df['Adj Close'][ticker].ewm(span=50, adjust=False).mean().values
            ema100 = df['Adj Close'][ticker].ewm(span=100, adjust=False).mean().values
            ema200 = df['Adj Close'][ticker].ewm(span=200, adjust=False).mean().values
            
            # MACD
            ema12 = df['Adj Close'][ticker].ewm(span=12, adjust=False).mean().values
            ema26 = df['Adj Close'][ticker].ewm(span=26, adjust=False).mean().values
            macd = ema12 - ema26
            signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
            
            # RSI (14)
            delta = pd.Series(adj_close).diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (20, 2)
            r_mean = pd.Series(adj_close).rolling(window=20).mean().values
            r_std = pd.Series(adj_close).rolling(window=20).std().values
            bb_upper = r_mean + 2 * r_std
            bb_lower = r_mean - 2 * r_std
            
            # Fill feature matrix
            state_matrix[:, idx, 0] = adj_close
            state_matrix[:, idx, 1] = volume
            state_matrix[:, idx, 2] = ema10
            state_matrix[:, idx, 3] = ema20
            state_matrix[:, idx, 4] = ema50
            state_matrix[:, idx, 5] = ema100
            state_matrix[:, idx, 6] = ema200
            state_matrix[:, idx, 7] = macd - signal
            state_matrix[:, idx, 8] = rsi
            state_matrix[:, idx, 9] = bb_upper
            state_matrix[:, idx, 10] = bb_lower
            state_matrix[:, idx, 11] = ret
            
        return np.nan_to_num(state_matrix)

    @staticmethod
    @njit
    def extract_har_features(returns_matrix: np.ndarray) -> np.ndarray:
        """
        returns_matrix: shape (T, N)
        Returns: har_features shape (T, N, 4) -> [1, r_d, r_w, r_m]
        """
        T, N = returns_matrix.shape
        har = np.zeros((T, N, 4))
        har[:, :, 0] = 1.0 # Bias constant
        
        for t in range(22, T):
            har[t, :, 1] = returns_matrix[t-1, :] # Daily (1 day)
            
            # Weekly average (5 days)
            har[t, :, 2] = np.sum(returns_matrix[t-5:t, :], axis=0) / 5.0
                
            # Monthly average (22 days)
            har[t, :, 3] = np.sum(returns_matrix[t-22:t, :], axis=0) / 22.0
        return har