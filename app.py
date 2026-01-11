import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

# Configurazione Pagina
st.set_page_config(page_title="NQL - AI Finance", layout="wide")

st.title("🧠 NQL: Neural Quant Lab")
st.markdown("Suite didattica per sperimentazione reti neurali LSTM su Apple Silicon M4.")

# --- BARRA LATERALE: CONFIGURAZIONE ---
st.sidebar.header("1. Dati")
ticker = st.sidebar.text_input("Ticker (es. AAPL, NVDA, BTC-USD)", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

st.sidebar.header("2. Rete Neurale")
look_back = st.sidebar.slider("Lookback (Giorni memoria)", 10, 120, 60)
epochs = st.sidebar.slider("Epoche (Training)", 1, 50, 10)
neurons = st.sidebar.slider("Neuroni", 10, 200, 50)

# --- FUNZIONI ---
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # Gestione caso in cui yfinance ritorni MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)
    
    # Se dopo xs i dati non hanno colonne standard, resettiamo
    if 'Close' not in data.columns and not data.empty:
         # Fallback semplice: prendiamo la prima colonna se c'è ambiguità
         pass 
         
    return data

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# --- CORE ---
if st.sidebar.button("Esegui NQL"):
    
    st.subheader(f"Analisi: {ticker}")
    with st.spinner('Download dati...'):
        df = load_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error("Errore: Nessun dato trovato.")
        else:
            # Assicuriamoci di avere la colonna Close pulita
            if 'Close' in df.columns:
                close_data = df[['Close']]
            else:
                # Se yfinance cambia formato, prendiamo la prima colonna utile
                close_data = df.iloc[:, 0:1] 

            st.line_chart(close_data)

            # PREPARAZIONE
            dataset = close_data.values
            training_data_len = int(np.ceil(len(dataset) * .80))

            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)

            train_data = scaled_data[0:int(training_data_len), :]
            x_train, y_train = create_dataset(train_data, look_back)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # TRAINING
            st.write(f"Addestramento su {len(x_train)} campioni...")
            
            model = Sequential()
            model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(neurons, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            prog_bar = st.progress(0)
            
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    prog_bar.progress((epoch + 1) / epochs)

            model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=[StreamlitCallback()], verbose=0)
            st.success("Training Completato!")

            # TEST & PREVISIONE
            test_data = scaled_data[training_data_len - look_back: , :]
            x_test, y_test = create_dataset(test_data, look_back)
            y_test = dataset[training_data_len:, :]
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            st.metric("Errore RMSE", f"{rmse:.2f}")

            # GRAFICO RISULTATI
            valid = close_data[training_data_len:].copy()
            valid['Predictions'] = predictions
            
            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(close_data[:training_data_len]['Close'], label='Training Data')
            ax.plot(valid['Close'], label='Dati Reali')
            ax.plot(valid['Predictions'], label='Previsione NQL')
            ax.legend()
            st.pyplot(fig)