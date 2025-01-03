from binance.client import Client
from binance.enums import *
import time
from config import API_KEY, API_SECRET, SYMBOL, TRADE_QUANTITY, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE

class Trader:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        self.position = None

    def get_current_price(self):
        """Obtém o preço atual do Bitcoin"""
        ticker = self.client.get_symbol_ticker(symbol=SYMBOL.replace('/', ''))
        return float(ticker['price'])

    def place_buy_order(self, price):
        """Coloca uma ordem de compra"""
        try:
            order = self.client.create_order(
                symbol=SYMBOL.replace('/', ''),
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=TRADE_QUANTITY
            )
            
            # Define stop loss e take profit
            stop_loss_price = price * (1 - STOP_LOSS_PERCENTAGE)
            take_profit_price = price * (1 + TAKE_PROFIT_PERCENTAGE)
            
            # Coloca ordens de stop loss e take profit
            self.client.create_order(
                symbol=SYMBOL.replace('/', ''),
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                quantity=TRADE_QUANTITY,
                price=str(stop_loss_price),
                stopPrice=str(stop_loss_price)
            )
            
            self.client.create_order(
                symbol=SYMBOL.replace('/', ''),
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                quantity=TRADE_QUANTITY,
                price=str(take_profit_price)
            )
            
            self.position = 'LONG'
            return order
        except Exception as e:
            print(f"Erro ao executar ordem de compra: {e}")
            return None

    def place_sell_order(self):
        """Coloca uma ordem de venda"""
        try:
            order = self.client.create_order(
                symbol=SYMBOL.replace('/', ''),
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=TRADE_QUANTITY
            )
            self.position = None
            return order
        except Exception as e:
            print(f"Erro ao executar ordem de venda: {e}")
            return None

    def check_position(self):
        """Verifica a posição atual"""
        return self.position
