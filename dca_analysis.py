import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class DCAAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance()
        
    def fetch_historical_data(self, days=730):
        """Busca dados históricos dos últimos X dias"""
        print(f"Coletando {days} dias de dados históricos...")
        
        symbol = 'BTC/USDT'
        timeframe = '1d'
        
        # Calcula timestamps
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Coleta dados
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe,
            int(start.timestamp() * 1000),
            limit=days
        )
        
        # Converte para DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def analyze_dca(self, df, weekly_investment=200):
        """Analisa estratégia DCA"""
        # Reamostra para dados semanais
        weekly_df = df.resample('W-MON').first().dropna()
        
        # Inicializa listas para armazenar dados
        dca_entries = []
        total_invested = 0
        total_btc = 0
        
        # Calcula compras semanais
        for date, row in weekly_df.iterrows():
            if not pd.isna(row['close']):
                btc_bought = weekly_investment / row['close']
                total_invested += weekly_investment
                total_btc += btc_bought
                
                dca_entries.append({
                    'date': date,
                    'price': row['close'],
                    'btc_bought': btc_bought,
                    'usd_invested': weekly_investment,
                    'total_invested': total_invested,
                    'total_btc': total_btc,
                    'portfolio_value': total_btc * row['close']
                })
        
        dca_df = pd.DataFrame(dca_entries)
        
        # Calcula métricas finais
        final_price = df['close'].iloc[-1]
        portfolio_value = total_btc * final_price
        total_return = ((portfolio_value - total_invested) / total_invested) * 100
        avg_price = total_invested / total_btc
        
        # Imprime relatório
        print("\n=== Relatório DCA Bitcoin ===")
        print(f"Período analisado: {df.index[0].strftime('%d/%m/%Y')} até {df.index[-1].strftime('%d/%m/%Y')}")
        print(f"\nInvestimento semanal: ${weekly_investment}")
        print(f"Total investido: ${total_invested:,.2f}")
        print(f"Bitcoin acumulado: {total_btc:.8f} BTC")
        print(f"Valor atual do portfólio: ${portfolio_value:,.2f}")
        print(f"Retorno total: {total_return:.2f}%")
        print(f"Preço médio de compra: ${avg_price:,.2f}")
        
        # Plota gráficos
        self.plot_dca_analysis(df, dca_df)
        
        return dca_df
    
    def plot_dca_analysis(self, price_df, dca_df):
        """Cria visualizações da análise DCA"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        
        # 1. Preço e Pontos de Compra
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(price_df.index, price_df['close'], label='Preço BTC', color='blue', alpha=0.7)
        ax1.scatter(dca_df['date'], dca_df['price'], 
                   color='green', alpha=0.5, s=50, label='Compras DCA')
        ax1.set_title('Preço do Bitcoin e Pontos de Compra DCA', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Investido vs. Valor do Portfólio
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dca_df['date'], dca_df['total_invested'], 
                label='Total Investido', color='red')
        ax2.plot(dca_df['date'], dca_df['portfolio_value'], 
                label='Valor do Portfólio', color='green')
        ax2.fill_between(dca_df['date'], 
                        dca_df['total_invested'], 
                        dca_df['portfolio_value'],
                        where=(dca_df['portfolio_value'] >= dca_df['total_invested']),
                        color='green', alpha=0.3)
        ax2.fill_between(dca_df['date'], 
                        dca_df['total_invested'], 
                        dca_df['portfolio_value'],
                        where=(dca_df['portfolio_value'] < dca_df['total_invested']),
                        color='red', alpha=0.3)
        ax2.set_title('Total Investido vs. Valor do Portfólio', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bitcoin Acumulado
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(dca_df['date'], dca_df['total_btc'], 
                label='BTC Acumulado', color='orange')
        ax3.fill_between(dca_df['date'], 0, dca_df['total_btc'], 
                        color='orange', alpha=0.3)
        ax3.set_title('Bitcoin Acumulado ao Longo do Tempo', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Preço Médio de Compra
        ax4 = fig.add_subplot(gs[2, 0])
        avg_prices = dca_df['total_invested'] / dca_df['total_btc']
        ax4.plot(dca_df['date'], avg_prices, 
                label='Preço Médio', color='purple')
        ax4.plot(dca_df['date'], dca_df['price'], 
                label='Preço de Mercado', color='gray', alpha=0.5)
        ax4.set_title('Evolução do Preço Médio de Compra', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Retorno Percentual
        ax5 = fig.add_subplot(gs[2, 1])
        returns = ((dca_df['portfolio_value'] - dca_df['total_invested']) / 
                  dca_df['total_invested'] * 100)
        ax5.plot(dca_df['date'], returns, label='Retorno %', color='blue')
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax5.fill_between(dca_df['date'], 0, returns,
                        where=(returns >= 0), color='green', alpha=0.3)
        ax5.fill_between(dca_df['date'], 0, returns,
                        where=(returns < 0), color='red', alpha=0.3)
        ax5.set_title('Retorno Percentual ao Longo do Tempo', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dca_analysis.png', dpi=300, bbox_inches='tight')
        print("\nGráfico de análise DCA salvo como 'dca_analysis.png'")
        plt.close()

def main():
    analyzer = DCAAnalyzer()
    df = analyzer.fetch_historical_data(days=730)  # 2 anos
    analyzer.analyze_dca(df, weekly_investment=200)

if __name__ == "__main__":
    main()
