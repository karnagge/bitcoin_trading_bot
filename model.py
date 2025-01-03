import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from config import FEATURES, LOOKBACK_PERIOD

class TradingModel:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = MinMaxScaler()

    def _build_model(self):
        """Constrói o modelo de rede neural LSTM"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(LOOKBACK_PERIOD, len(FEATURES))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, df):
        """Prepara os dados para treinamento"""
        # Seleciona apenas as features relevantes
        data = df[FEATURES].values
        
        # Normaliza os dados
        data_normalized = self.scaler.fit_transform(data)
        
        # Prepara as sequências para o LSTM
        X, y = [], []
        for i in range(len(data_normalized) - LOOKBACK_PERIOD):
            X.append(data_normalized[i:(i + LOOKBACK_PERIOD)])
            # Define o target como 1 se o preço subiu, 0 se caiu
            price_direction = 1 if data[i + LOOKBACK_PERIOD][0] > data[i + LOOKBACK_PERIOD - 1][0] else 0
            y.append(price_direction)
            
        return np.array(X), np.array(y)

    def train(self, X, y, epochs=50, batch_size=32):
        """Treina o modelo"""
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, data):
        """Faz previsões com o modelo treinado"""
        # Prepara os dados para previsão
        data_normalized = self.scaler.transform(data)
        data_sequence = np.array([data_normalized[-LOOKBACK_PERIOD:]])
        
        # Faz a previsão
        prediction = self.model.predict(data_sequence)
        return prediction[0][0]  # Retorna a probabilidade de subida do preço
