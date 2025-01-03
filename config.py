import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações da API Binance
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Configurações de trading
SYMBOL = 'BTC/USDT'
TRADE_QUANTITY = 0.001  # Quantidade mínima de BTC para trade
STOP_LOSS_PERCENTAGE = 0.02  # 2%
TAKE_PROFIT_PERCENTAGE = 0.03  # 3%

# Configurações do modelo de IA
LOOKBACK_PERIOD = 365  # Período de análise em dias (1 ano)
HISTORICAL_DATA_DAYS = 730  # 2 anos de dados históricos
FEATURES = ['close', 'volume', 'rsi', 'macd', 'macd_hist', 'bollinger_upper', 'bollinger_lower', 
            'sma_50', 'sma_200', 'momentum', 'atr']

# Configurações de sinais
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_THRESHOLD = 0
PRICE_CHANGE_THRESHOLD = 0.02  # 2% de mudança
VOLUME_INCREASE_THRESHOLD = 1.5  # 50% de aumento no volume
