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
        df['signal_reasons'] = ''  # Armazena as razões para cada sinal
        last_signal = 0  # Controle para alternar sinais
        
        for i in range(1, len(df)):
            # Condições de Compra
            buy_conditions = {
                'RSI_OVERSOLD': df.iloc[i]['rsi'] < RSI_OVERSOLD,
                'MACD_CROSS_UP': df.iloc[i-1]['macd'] < df.iloc[i-1]['signal'] and df.iloc[i]['macd'] > df.iloc[i]['signal'],
                'PRICE_NEAR_BB_LOW': df.iloc[i]['close'] <= df.iloc[i]['bollinger_lower'] * 1.02,
                'VOLUME_INCREASE': df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * VOLUME_INCREASE_THRESHOLD,
                'UPTREND': df.iloc[i]['sma_50'] > df.iloc[i]['sma_200']
            }
            
            # Condições de Venda
            sell_conditions = {
                'RSI_OVERBOUGHT': df.iloc[i]['rsi'] > RSI_OVERBOUGHT,
                'MACD_CROSS_DOWN': df.iloc[i-1]['macd'] > df.iloc[i-1]['signal'] and df.iloc[i]['macd'] < df.iloc[i]['signal'],
                'PRICE_NEAR_BB_HIGH': df.iloc[i]['close'] >= df.iloc[i]['bollinger_upper'] * 0.98,
                'VOLUME_DECREASE': df.iloc[i]['volume'] < df.iloc[i-1]['volume'] * 0.7,
                'DOWNTREND': df.iloc[i]['sma_50'] < df.iloc[i]['sma_200']
            }
            
            # Verifica condições de compra apenas se o último sinal foi venda ou neutro
            if last_signal <= 0:
                active_buy_conditions = [cond for cond, is_true in buy_conditions.items() if is_true]
                if len(active_buy_conditions) >= 3:
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    df.iloc[i, df.columns.get_loc('signal_reasons')] = 'COMPRA: ' + ', '.join(active_buy_conditions)
                    last_signal = 1
            
            # Verifica condições de venda apenas se o último sinal foi compra
            elif last_signal == 1:
                active_sell_conditions = [cond for cond, is_true in sell_conditions.items() if is_true]
                if len(active_sell_conditions) >= 3:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    df.iloc[i, df.columns.get_loc('signal_reasons')] = 'VENDA: ' + ', '.join(active_sell_conditions)
                    last_signal = -1
        
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
        # Configuração do estilo
        plt.style.use('seaborn')
        
        # Cria uma figura com subplots em grid
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Gráfico de Preço e Sinais
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['close'], label='Preço', alpha=0.7, color='blue')
        ax1.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.5, color='orange')
        ax1.plot(df.index, df['sma_200'], label='SMA 200', alpha=0.5, color='red')
        
        # Adiciona Bandas de Bollinger
        ax1.plot(df.index, df['bollinger_upper'], '--', color='gray', alpha=0.3)
        ax1.plot(df.index, df['bollinger_lower'], '--', color='gray', alpha=0.3)
        ax1.fill_between(df.index, df['bollinger_upper'], df['bollinger_lower'], alpha=0.1, color='gray')
        
        # Adiciona sinais com anotações
        for idx, row in df[df['signal'] != 0].iterrows():
            color = 'green' if row['signal'] == 1 else 'red'
            marker = '^' if row['signal'] == 1 else 'v'
            ax1.scatter(idx, row['close'], color=color, marker=marker, s=100)
            
            # Adiciona anotação com as razões
            ax1.annotate(row['signal_reasons'], 
                        xy=(idx, row['close']),
                        xytext=(10, 10 if row['signal'] == 1 else -10),
                        textcoords='offset points',
                        ha='left',
                        va='bottom' if row['signal'] == 1 else 'top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        rotation=45)
        
        ax1.set_title('Preço com Sinais de Trading e Médias Móveis')
        ax1.legend()
        
        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
        ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
        ax2.set_title('RSI')
        ax2.set_ylim(0, 100)
        
        # 3. MACD
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(df.index, df['macd'], label='MACD', color='blue')
        ax3.plot(df.index, df['signal'], label='Signal', color='orange')
        ax3.bar(df.index, df['macd_hist'], label='Histograma', color='gray', alpha=0.3)
        ax3.set_title('MACD')
        ax3.legend()
        
        # 4. Volume
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.bar(df.index, df['volume'], label='Volume', color='blue', alpha=0.3)
        ax4.set_title('Volume')
        
        # 5. Momentum
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(df.index, df['momentum'], label='Momentum', color='blue')
        ax5.fill_between(df.index, df['momentum'], 0, 
                        where=(df['momentum'] >= 0), color='green', alpha=0.3)
        ax5.fill_between(df.index, df['momentum'], 0, 
                        where=(df['momentum'] < 0), color='red', alpha=0.3)
        ax5.set_title('Momentum')
        
        plt.tight_layout()
        plt.savefig('strategy_analysis.png', dpi=300, bbox_inches='tight')
        print("\nGráfico de análise detalhado salvo como 'strategy_analysis.png'")
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
