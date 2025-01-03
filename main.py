from data_collector import DataCollector
from model import TradingModel
from trader import Trader
import time
import schedule

def trading_job():
    """Função principal de trading"""
    try:
        # Inicializa as classes
        collector = DataCollector()
        model = TradingModel()
        trader = Trader()

        # Coleta dados históricos
        df = collector.fetch_ohlcv_data()
        if df is None:
            print("Erro ao coletar dados. Tentando novamente no próximo ciclo.")
            return

        # Calcula indicadores
        df = collector.calculate_indicators(df)
        
        # Prepara dados para o modelo
        X, y = model.prepare_data(df)
        
        # Treina o modelo (em produção, você pode querer fazer isso separadamente)
        model.train(X, y)
        
        # Obtém previsão para o próximo movimento
        current_data = df.iloc[-1][['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']].values.reshape(1, -1)
        prediction = model.predict(current_data)
        
        current_position = trader.check_position()
        current_price = trader.get_current_price()

        # Lógica de trading
        if prediction > 0.7 and current_position is None:  # Sinal de compra forte
            print(f"Sinal de compra detectado. Probabilidade: {prediction:.2f}")
            trader.place_buy_order(current_price)
        elif prediction < 0.3 and current_position == 'LONG':  # Sinal de venda forte
            print(f"Sinal de venda detectado. Probabilidade: {prediction:.2f}")
            trader.place_sell_order()
        else:
            print(f"Nenhuma ação necessária. Probabilidade: {prediction:.2f}")

    except Exception as e:
        print(f"Erro no ciclo de trading: {e}")

def main():
    print("Iniciando bot de trading...")
    
    # Agenda a execução do job a cada 1 hora
    schedule.every(1).hours.do(trading_job)
    
    # Executa o primeiro job imediatamente
    trading_job()
    
    # Loop principal
    while True:
        schedule.run_pending()
        time.sleep(60)  # Espera 1 minuto antes de verificar novamente

if __name__ == "__main__":
    main()
