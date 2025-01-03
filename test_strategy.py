from data_collector import DataCollector
from model import TradingModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class StrategyTester:
    def __init__(self):
        self.collector = DataCollector()
        self.model = TradingModel()
        
    def analyze_signals(self, timeframe='1h', limit=500):
        """Analisa os sinais gerados pela estratégia"""
        # Coleta dados históricos
        print("Coletando dados históricos...")
        df = self.collector.fetch_ohlcv_data(timeframe=timeframe, limit=limit)
        df = self.collector.calculate_indicators(df)
        
        # Prepara dados para o modelo
        print("Preparando dados e treinando modelo...")
        X, y = self.model.prepare_data(df)
        self.model.train(X, y, epochs=20)
        
        # Gera previsões
        predictions = []
        for i in range(len(df) - self.model.model.input_shape[1]):
            data = df.iloc[i:i+self.model.model.input_shape[1]][['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']]
            pred = self.model.predict(data.values)
            predictions.append(pred)
        
        # Adiciona previsões ao DataFrame
        df['prediction'] = pd.Series(predictions + [np.nan] * (len(df) - len(predictions)))
        
        # Gera sinais
        df['signal'] = 0  # 0: Neutro, 1: Compra, -1: Venda
        df.loc[df['prediction'] > 0.7, 'signal'] = 1
        df.loc[df['prediction'] < 0.3, 'signal'] = -1
        
        return self.analyze_results(df)
    
    def analyze_results(self, df):
        """Analisa os resultados da estratégia"""
        results = {
            'total_signals': len(df[df['signal'] != 0]),
            'buy_signals': len(df[df['signal'] == 1]),
            'sell_signals': len(df[df['signal'] == -1]),
            'signal_distribution': df['signal'].value_counts(),
            'avg_rsi_buy': df[df['signal'] == 1]['rsi'].mean(),
            'avg_rsi_sell': df[df['signal'] == -1]['rsi'].mean(),
            'success_rate': self.calculate_success_rate(df)
        }
        
        # Plota gráficos
        self.plot_analysis(df)
        
        return results
    
    def calculate_success_rate(self, df):
        """Calcula taxa de sucesso dos sinais"""
        success = 0
        total = 0
        
        for i in range(len(df) - 1):
            if df.iloc[i]['signal'] != 0:
                total += 1
                current_price = df.iloc[i]['close']
                next_price = df.iloc[i + 1]['close']
                
                if df.iloc[i]['signal'] == 1:  # Sinal de compra
                    if next_price > current_price:
                        success += 1
                else:  # Sinal de venda
                    if next_price < current_price:
                        success += 1
        
        return (success / total * 100) if total > 0 else 0
    
    def plot_analysis(self, df):
        """Plota gráficos de análise"""
        plt.figure(figsize=(15, 10))
        
        # Preço e sinais
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['close'], label='Preço', alpha=0.7)
        plt.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['close'], 
                   color='green', marker='^', label='Compra')
        plt.scatter(df[df['signal'] == -1].index, df[df['signal'] == -1]['close'], 
                   color='red', marker='v', label='Venda')
        plt.title('Preço e Sinais')
        plt.legend()
        
        # Distribuição de RSI
        plt.subplot(2, 2, 2)
        sns.histplot(data=df, x='rsi', hue=df['signal'].map({0: 'Neutro', 1: 'Compra', -1: 'Venda'}))
        plt.title('Distribuição de RSI por Sinal')
        
        # Correlação entre indicadores
        plt.subplot(2, 2, 3)
        correlation = df[['close', 'volume', 'rsi', 'macd', 'prediction']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlação entre Indicadores')
        
        # Distribuição de previsões
        plt.subplot(2, 2, 4)
        sns.histplot(data=df, x='prediction', bins=30)
        plt.title('Distribuição das Previsões')
        
        plt.tight_layout()
        plt.savefig('strategy_analysis.png')
        plt.close()

def main():
    tester = StrategyTester()
    print("Iniciando análise da estratégia...")
    
    # Testa diferentes timeframes
    timeframes = ['1h', '4h', '1d']
    for timeframe in timeframes:
        print(f"\nAnalisando timeframe: {timeframe}")
        results = tester.analyze_signals(timeframe=timeframe)
        
        print("\nResultados da Análise:")
        print(f"Total de sinais: {results['total_signals']}")
        print(f"Sinais de compra: {results['buy_signals']}")
        print(f"Sinais de venda: {results['sell_signals']}")
        print(f"Taxa de sucesso: {results['success_rate']:.2f}%")
        print(f"RSI médio (compra): {results['avg_rsi_buy']:.2f}")
        print(f"RSI médio (venda): {results['avg_rsi_sell']:.2f}")
        print("\nDistribuição dos sinais:")
        print(results['signal_distribution'])
        
        print("\nGráficos salvos em 'strategy_analysis.png'")

if __name__ == "__main__":
    main()