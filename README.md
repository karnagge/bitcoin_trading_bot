# Bot de Trading de Bitcoin com IA

Este é um bot automatizado de trading de Bitcoin que utiliza Inteligência Artificial para tomar decisões de compra e venda.

## Características

- Coleta automática de dados históricos da Binance
- Análise técnica com múltiplos indicadores (RSI, MACD, Bollinger Bands)
- Modelo de IA baseado em LSTM para previsão de movimentos de preço
- Sistema automatizado de trading com gestão de risco
- Stop Loss e Take Profit automáticos

## Requisitos

- Python 3.8+
- Conta na Binance com API Key e Secret Key
- Pacotes Python listados em `requirements.txt`

## Configuração

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Crie um arquivo `.env` na raiz do projeto com suas credenciais da Binance:
```
BINANCE_API_KEY=sua_api_key_aqui
BINANCE_API_SECRET=sua_api_secret_aqui
```

4. Ajuste as configurações em `config.py` conforme necessário

## Uso

Para iniciar o bot:
```bash
python main.py
```

## Aviso de Risco

Este bot é apenas para fins educacionais. Trading de criptomoedas envolve riscos significativos. Use por sua conta e risco.

## Estrutura do Projeto

- `main.py`: Arquivo principal que executa o bot
- `config.py`: Configurações do bot
- `data_collector.py`: Coleta dados da Binance
- `model.py`: Implementação do modelo de IA
- `trader.py`: Execução das operações de trading

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
