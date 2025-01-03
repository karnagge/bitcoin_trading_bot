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
            # Verifica se o preço está acima da média móvel de 200 dias
            above_200ma = df.iloc[i]['close'] > df.iloc[i]['sma_200']
            
            # Condições de Compra
            buy_conditions = {
                'ABOVE_200MA': above_200ma,
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
                # Só gera sinal de compra se estiver acima da média móvel de 200 dias
                if 'ABOVE_200MA' in active_buy_conditions and len(active_buy_conditions) >= 4:  # Aumentei para 4 pois ABOVE_200MA é obrigatório
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
        total_trades = len(df[df['signal'] != 0])
        buy_signals = len(df[df['signal'] == 1])
        sell_signals = len(df[df['signal'] == -1])
        
        print("\n=== Relatório de Trading ===")
        print(f"Total de sinais: {total_trades}")
        print(f"Sinais de compra: {buy_signals}")
        print(f"Sinais de venda: {sell_signals}")
        
        # Calcula lucro/prejuízo
        trades = []
        current_position = None
        entry_price = 0
        total_profit = 0
        winning_trades = 0
        losing_trades = 0
        
        for idx, row in df.iterrows():
            if row['signal'] == 1 and current_position is None:  # Compra
                current_position = 'long'
                entry_price = row['close']
            elif row['signal'] == -1 and current_position == 'long':  # Venda
                exit_price = row['close']
                profit = ((exit_price - entry_price) / entry_price) * 100
                total_profit += profit
                trades.append(profit)
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                current_position = None
        
        if trades:
            avg_profit = sum(trades) / len(trades)
            win_rate = (winning_trades / len(trades)) * 100
            
            print("\n=== Análise de Performance ===")
            print(f"Total de trades completados: {len(trades)}")
            print(f"Lucro/Prejuízo total: {total_profit:.2f}%")
            print(f"Média de lucro por trade: {avg_profit:.2f}%")
            print(f"Taxa de acerto: {win_rate:.2f}%")
            print(f"Trades vencedores: {winning_trades}")
            print(f"Trades perdedores: {losing_trades}")
            
            if trades:
                print("\nMelhores trades:")
                best_trades = sorted(trades, reverse=True)[:3]
                for i, trade in enumerate(best_trades, 1):
                    print(f"{i}. {trade:.2f}%")
                
                print("\nPiores trades:")
                worst_trades = sorted(trades)[:3]
                for i, trade in enumerate(worst_trades, 1):
                    print(f"{i}. {trade:.2f}%")
        
        print("\n=== Comparativo com Buy and Hold (DCA) ===")
        
        # Simulação de DCA (Dollar Cost Averaging)
        weekly_investment = 200  # USD
        total_invested_dca = 0
        total_btc_dca = 0
        dca_entries = []
        
        # Agrupa os dados por semana e pega o primeiro preço de cada semana
        df['week'] = pd.to_datetime(df.index).isocalendar().week
        df['year'] = pd.to_datetime(df.index).year
        weekly_prices = df.groupby(['year', 'week'])['close'].first()
        
        # Calcula DCA
        for price in weekly_prices:
            btc_bought = weekly_investment / price
            total_invested_dca += weekly_investment
            total_btc_dca += btc_bought
            dca_entries.append({
                'price': price,
                'btc_bought': btc_bought,
                'usd_invested': weekly_investment
            })
        
        # Calcula resultado final do DCA
        final_price = df['close'].iloc[-1]
        dca_final_value = total_btc_dca * final_price
        dca_profit_pct = ((dca_final_value - total_invested_dca) / total_invested_dca) * 100
        
        print("\n=== Resultados DCA ($200/semana) ===")
        print(f"Total investido: ${total_invested_dca:,.2f}")
        print(f"Total BTC acumulado: {total_btc_dca:.8f}")
        print(f"Valor final: ${dca_final_value:,.2f}")
        print(f"Lucro/Prejuízo: {dca_profit_pct:.2f}%")
        print(f"Preço médio de compra: ${(total_invested_dca/total_btc_dca):,.2f}")
        
        # Compara com a estratégia de trading
        if trades:
            print("\n=== Comparativo de Estratégias ===")
            print(f"Trading Bot: {total_profit:.2f}%")
            print(f"DCA: {dca_profit_pct:.2f}%")
            print(f"Diferença: {(total_profit - dca_profit_pct):.2f}%")
        
        # Adiciona gráfico comparativo
        self.plot_comparison(df, dca_entries, total_profit, dca_profit_pct)
        
        self.plot_analysis(df)
        return df
    
    def plot_comparison(self, df, dca_entries, trading_profit, dca_profit):
        """Plota gráfico comparativo entre as estratégias"""
        plt.figure(figsize=(15, 8))
        
        # Preço do Bitcoin
        plt.plot(df.index, df['close'], label='Preço BTC', color='blue', alpha=0.5)
        
        # Marca pontos de compra DCA
        weekly_data = pd.DataFrame(dca_entries)
        plt.scatter(df.groupby(['year', 'week']).first().index, 
                   weekly_data['price'],
                   color='green', alpha=0.2, s=30, label='Compras DCA')
        
        # Marca sinais de trading
        plt.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['close'], 
                   color='green', marker='^', s=100, label='Compra (Trading)')
        plt.scatter(df[df['signal'] == -1].index, df[df['signal'] == -1]['close'], 
                   color='red', marker='v', s=100, label='Venda (Trading)')
        
        # Adiciona legendas com resultados
        plt.title('Comparativo: Trading Bot vs DCA ($200/semana)')
        plt.text(0.02, 0.98, f'Trading Bot: {trading_profit:.2f}%\nDCA: {dca_profit:.2f}%', 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salva o gráfico comparativo
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        print("\nGráfico comparativo salvo como 'strategy_comparison.png'")
        plt.close()

    def plot_analysis(self, df):
        """Plota gráficos de análise"""
        # Configuração do estilo
        plt.rcParams['figure.figsize'] = [20, 15]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Cria uma figura com subplots em grid
        fig = plt.figure()
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        
        # 1. Gráfico de Preço e Sinais
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['close'], label='Preço', alpha=0.7, color='blue', linewidth=2)
        ax1.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.5, color='orange', linewidth=1)
        ax1.plot(df.index, df['sma_200'], label='SMA 200', alpha=0.5, color='red', linewidth=1)
        
        # Adiciona Bandas de Bollinger
        ax1.plot(df.index, df['bollinger_upper'], '--', color='gray', alpha=0.3)
        ax1.plot(df.index, df['bollinger_lower'], '--', color='gray', alpha=0.3)
        ax1.fill_between(df.index, df['bollinger_upper'], df['bollinger_lower'], alpha=0.1, color='gray')
        
        # Adiciona sinais com anotações
        for idx, row in df[df['signal'] != 0].iterrows():
            color = 'green' if row['signal'] == 1 else 'red'
            marker = '^' if row['signal'] == 1 else 'v'
            ax1.scatter(idx, row['close'], color=color, marker=marker, s=200, zorder=5)
            
            # Adiciona anotação com as razões
            ax1.annotate(row['signal_reasons'], 
                        xy=(idx, row['close']),
                        xytext=(20, 20 if row['signal'] == 1 else -20),
                        textcoords='offset points',
                        ha='left',
                        va='bottom' if row['signal'] == 1 else 'top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        rotation=45,
                        fontsize=8)
        
        ax1.set_title('Preço com Sinais de Trading e Médias Móveis', fontsize=12, pad=20)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
        ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
        ax2.set_title('RSI (Relative Strength Index)', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1)
        ax3.plot(df.index, df['signal'], label='Signal', color='orange', linewidth=1)
        ax3.bar(df.index, df['macd_hist'], label='Histograma', color='gray', alpha=0.3)
        ax3.set_title('MACD (Moving Average Convergence Divergence)', fontsize=10)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume
        ax4 = fig.add_subplot(gs[2, 0])
        volume_colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
        ax4.bar(df.index, df['volume'], color=volume_colors, alpha=0.5)
        ax4.set_title('Volume', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Momentum
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(df.index, df['momentum'], label='Momentum', color='blue', linewidth=1)
        ax5.fill_between(df.index, df['momentum'], 0, 
                        where=(df['momentum'] >= 0), color='green', alpha=0.3)
        ax5.fill_between(df.index, df['momentum'], 0, 
                        where=(df['momentum'] < 0), color='red', alpha=0.3)
        ax5.set_title('Momentum', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Ajusta o layout
        plt.tight_layout()
        
        # Salva o gráfico
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
        print(f"Total de sinais: {len(results[results['signal'] != 0])}")
        print(f"Sinais de compra: {len(results[results['signal'] == 1])}")
        print(f"Sinais de venda: {len(results[results['signal'] == -1])}")
        print(f"Taxa de sucesso: {tester.calculate_success_rate(results):.2f}%")
        print(f"RSI médio (compra): {results[results['signal'] == 1]['rsi'].mean():.2f}")
        print(f"RSI médio (venda): {results[results['signal'] == -1]['rsi'].mean():.2f}")
        print("\nDistribuição dos sinais:")
        print(results['signal'].value_counts())
        
        print("\nGráficos salvos em 'strategy_analysis.png'")

if __name__ == "__main__":
    main()
