import unittest
from data_collector import DataCollector
from model import TradingModel
from trader import Trader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestBitcoinBot(unittest.TestCase):
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.collector = DataCollector()
        self.model = TradingModel()
        self.trader = Trader()

    def test_data_collection(self):
        """Testa a coleta de dados"""
        df = self.collector.fetch_ohlcv_data(limit=100)
        
        # Verifica se os dados foram coletados
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Verifica se todas as colunas necessárias estão presentes
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)

    def test_indicators(self):
        """Testa o cálculo dos indicadores técnicos"""
        df = self.collector.fetch_ohlcv_data(limit=100)
        df = self.collector.calculate_indicators(df)
        
        # Verifica se os indicadores foram calculados
        indicators = ['rsi', 'macd', 'signal', 'bollinger_upper', 'bollinger_lower']
        for indicator in indicators:
            self.assertIn(indicator, df.columns)
            
        # Verifica se os valores dos indicadores estão dentro dos ranges esperados
        self.assertTrue(all(df['rsi'].between(0, 100)))
        self.assertTrue(all(df['bollinger_upper'] > df['bollinger_lower']))

    def test_model_training(self):
        """Testa o treinamento do modelo"""
        # Prepara dados de teste
        df = self.collector.fetch_ohlcv_data(limit=100)
        df = self.collector.calculate_indicators(df)
        
        X, y = self.model.prepare_data(df)
        
        # Verifica se os dados foram preparados corretamente
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        
        # Treina o modelo
        history = self.model.train(X, y, epochs=5)
        
        # Verifica se o treinamento foi concluído
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

    def test_model_prediction(self):
        """Testa as previsões do modelo"""
        # Prepara e treina o modelo
        df = self.collector.fetch_ohlcv_data(limit=100)
        df = self.collector.calculate_indicators(df)
        X, y = self.model.prepare_data(df)
        self.model.train(X, y, epochs=5)
        
        # Testa previsão
        current_data = df.iloc[-1][['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']].values.reshape(1, -1)
        prediction = self.model.predict(current_data)
        
        # Verifica se a previsão está no formato esperado
        self.assertTrue(0 <= prediction <= 1)

    def test_trader_functions(self):
        """Testa as funções básicas do trader"""
        # Testa obtenção do preço atual
        price = self.trader.get_current_price()
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)
        
        # Testa verificação de posição
        position = self.trader.check_position()
        self.assertIn(position, [None, 'LONG'])

def run_tests():
    """Executa todos os testes"""
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
