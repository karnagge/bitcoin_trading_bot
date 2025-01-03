from data_collector import DataCollector
from model import TradingModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_balance = 0
        self.trades = []
        self.collector = DataCollector()
        self.model = TradingModel()

    def run_backtest(self, start_date, end_date, timeframe='1d'):
        """Executa o backtesting no período especificado"""
        # Coleta dados históricos
        df = self.collector.fetch_ohlcv_data(timeframe=timeframe, limit=1000)
        df = self.collector.calculate_indicators(df)
        
        # Prepara e treina o modelo com dados anteriores ao período de teste
        train_data = df[df['timestamp'] < start_date]
        X_train, y_train = self.model.prepare_data(train_data)
        self.model.train(X_train, y_train, epochs=20)
        
        # Filtra dados para o período de teste
        test_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Simula trading
        for i in range(len(test_data) - 1):
            current_data = test_data.iloc[i]
            next_data = test_data.iloc[i + 1]
            
            # Prepara dados para previsão
            features = current_data[['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']].values.reshape(1, -1)
            prediction = self.model.predict(features)
            
            # Simula decisões de trading
            if prediction > 0.7 and self.btc_balance == 0:
                # Compra
                price = current_data['close']
                amount = (self.balance * 0.95) / price  # Usa 95% do saldo disponível
                self.btc_balance = amount
                self.balance -= amount * price
                self.trades.append({
                    'timestamp': current_data['timestamp'],
                    'type': 'BUY',
                    'price': price,
                    'amount': amount,
                    'balance': self.balance + (self.btc_balance * price)
                })
                
            elif prediction < 0.3 and self.btc_balance > 0:
                # Vende
                price = current_data['close']
                amount = self.btc_balance
                self.balance += amount * price
                self.btc_balance = 0
                self.trades.append({
                    'timestamp': current_data['timestamp'],
                    'type': 'SELL',
                    'price': price,
                    'amount': amount,
                    'balance': self.balance
                })
        
        return self.calculate_statistics()
    
    def calculate_statistics(self):
        """Calcula estatísticas do backtest"""
        if not self.trades:
            return {
                'total_trades': 0,
                'profit_loss': 0,
                'return_percentage': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calcula o valor final (incluindo BTC não vendido ao último preço)
        final_balance = self.balance + (self.btc_balance * self.trades[-1]['price'])
        
        stats = {
            'total_trades': len(self.trades),
            'profit_loss': final_balance - self.initial_balance,
            'return_percentage': ((final_balance - self.initial_balance) / self.initial_balance) * 100,
            'winning_trades': len(trades_df[trades_df['type'] == 'SELL'][trades_df['balance'] > trades_df['balance'].shift(1)]),
            'average_trade_duration': trades_df['timestamp'].diff().mean() if len(trades_df) > 1 else 0
        }
        
        return stats

if __name__ == "__main__":
    # Exemplo de uso
    backtester = Backtester(initial_balance=10000)
    
    # Define período de teste (últimos 30 dias)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    results = backtester.run_backtest(start_date, end_date)
    
    print("\n=== Resultados do Backtesting ===")
    print(f"Total de trades: {results['total_trades']}")
    print(f"Lucro/Prejuízo: ${results['profit_loss']:.2f}")
    print(f"Retorno: {results['return_percentage']:.2f}%")
    print(f"Trades vencedores: {results['winning_trades']}")
    if results['average_trade_duration']:
        print(f"Duração média dos trades: {results['average_trade_duration']}")
