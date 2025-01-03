import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from config import API_KEY, API_SECRET, SYMBOL

class DataCollector:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True
        })

    def fetch_ohlcv_data(self, timeframe='1h', limit=100):
        """Coleta dados OHLCV (Open, High, Low, Close, Volume) do par BTC/USDT"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Erro ao coletar dados: {e}")
            return None

    def calculate_indicators(self, df):
        """Calcula indicadores técnicos"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['sma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma'] + (df['std'] * 2)
        df['bollinger_lower'] = df['sma'] - (df['std'] * 2)

        return df
