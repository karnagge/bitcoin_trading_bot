from data_collector import DataCollector
from model import TradingModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Constantes para RSI
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Constante para aumento de volume
VOLUME_INCREASE_THRESHOLD = 1.5

class StrategyTester:
    def __init__(self):
        self.collector = DataCollector()
        self.model = TradingModel()
        
    def analyze_signals(self, timeframe='1d', limit=730):
        """Analisa os sinais gerados pela estratégia"""
        # Coleta dados históricos
        print("Coletando dados históricos...")
        df = self.collector.fetch_ohlcv_data(timeframe=timeframe, limit=limit)  # 2 anos de dados
        df = self.collector.calculate_indicators(df)
        
        # Gera sinais baseados em múltiplos indicadores
        df['signal'] = 0  # 0: Neutro, 1: Compra, -1: Venda
        
        for i in range(1, len(df)):
            # Condições de Compra
            buy_conditions = [
                # RSI em sobrevenda
                df.iloc[i]['rsi'] < RSI_OVERSOLD,
                # MACD cruzando para cima
                df.iloc[i-1]['macd'] < df.iloc[i-1]['signal'] and df.iloc[i]['macd'] > df.iloc[i]['signal'],
                # Preço próximo à Bollinger inferior
                df.iloc[i]['close'] <= df.iloc[i]['bollinger_lower'] * 1.02,
                # Volume aumentando
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * VOLUME_INCREASE_THRESHOLD,
                # Tendência de alta
                df.iloc[i]['sma_50'] > df.iloc[i]['sma_200']
            ]
            
            # Condições de Venda
            sell_conditions = [
                # RSI em sobrecompra
                df.iloc[i]['rsi'] > RSI_OVERBOUGHT,
                # MACD cruzando para baixo
                df.iloc[i-1]['macd'] > df.iloc[i-1]['signal'] and df.iloc[i]['macd'] < df.iloc[i]['signal'],
                # Preço próximo à Bollinger superior
                df.iloc[i]['close'] >= df.iloc[i]['bollinger_upper'] * 0.98,
                # Volume diminuindo
                df.iloc[i]['volume'] < df.iloc[i-1]['volume'] * 0.7,
                # Tendência de baixa
                df.iloc[i]['sma_50'] < df.iloc[i]['sma_200']
            ]
            
            # Gera sinal apenas se múltiplas condições forem atendidas
            if sum(buy_conditions) >= 3:  # Pelo menos 3 condições de compra
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif sum(sell_conditions) >= 3:  # Pelo menos 3 condições de venda
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return self.analyze_results(df)
    
    def analyze_results(self, df):
        """Analisa os resultados da estratégia"""
        # Calcula estatísticas básicas
        total_signals = len(df[df['signal'] != 0])
        buy_signals = len(df[df['signal'] == 1])
        sell_signals = len(df[df['signal'] == -1])
        
        # Calcula retornos para sinais de compra
        returns = []
        for i in range(len(df)-1):
            if df.iloc[i]['signal'] == 1:  # Sinal de compra
                entry_price = df.iloc[i]['close']
                exit_price = df.iloc[i+1]['close']
                returns.append((exit_price - entry_price) / entry_price * 100)
        
        avg_return = np.mean(returns) if returns else 0
        
        results = {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_distribution': df['signal'].value_counts(),
            'avg_rsi_buy': df[df['signal'] == 1]['rsi'].mean(),
            'avg_rsi_sell': df[df['signal'] == -1]['rsi'].mean(),
            'success_rate': self.calculate_success_rate(df),
            'average_return': avg_return,
            'momentum_correlation': df['momentum'].corr(df['close']) if 'momentum' in df else 0
        }
        
        # Plota os gráficos
        self.plot_analysis(df)
        
        return results
    
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
        correlation = df[['close', 'volume', 'rsi', 'macd', 'macd_hist', 'sma_50', 'sma_200']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlação entre Indicadores')
        
        # Volume e Momentum
        plt.subplot(2, 2, 4)
        plt.plot(df.index, df['momentum'], label='Momentum', color='blue', alpha=0.7)
        plt.fill_between(df.index, df['momentum'], 0, where=(df['momentum'] >= 0), color='green', alpha=0.3)
        plt.fill_between(df.index, df['momentum'], 0, where=(df['momentum'] < 0), color='red', alpha=0.3)
        plt.title('Momentum ao Longo do Tempo')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('strategy_analysis.png')
        print("\nGráfico de análise salvo como 'strategy_analysis.png'")
        plt.close()

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
    
def main():
    tester = StrategyTester()
    print("Iniciando análise da estratégia...")
    
    # Testa diferentes timeframes
    timeframes = ['1d', '3d', '1w']
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
        print(f"Retorno médio: {results['average_return']:.2f}%")
        print(f"Correlação de momentum: {results['momentum_correlation']:.2f}")
        print("\nDistribuição dos sinais:")
        print(results['signal_distribution'])
        
        print("\nGráficos salvos em 'strategy_analysis.png'")

if __name__ == "__main__":
    main()
