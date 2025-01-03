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
LOOKBACK_PERIOD = 24  # Período de análise em horas
FEATURES = ['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
