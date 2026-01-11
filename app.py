import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

# --- LISTE PRE-CARICATE PER AUTOCOMPLETE (TOP STOCKS) ---
NASDAQ_TOP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST",
    "CSCO", "TMUS", "CMCSA", "ADBE", "NFLX", "TXN", "AMD", "QCOM", "INTC", "HON",
    "AMGN", "INTU", "SBUX", "GILD", "BKNG", "MDLZ", "ADI", "ISRG", "ADP", "REGN",
    "PYPL", "VRTX", "FISV", "LRCX", "ATVI", "MU", "MELI", "CSX", "PANW", "MRNA",
    "SNOPS", "ASML", "CDNS", "CHTR", "KLAC", "MAR", "ORLY", "MNST", "FTNT", "ABNB"
]

NYSE_TOP = [
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "XOM", "UNH", "HD", "BAC",
    "KO", "LLY", "CVX", "MRK", "ABBV", "PFE", "DIS", "T", "VZ", "NKE",
    "MCD", "WFC", "UPS", "BMY", "NEE", "BA", "PM", "MS", "RTX", "CAT",
    "GS", "IBM", "MMM", "GE", "C", "F", "GM", "DE", "LMT", "TGT",
    "LOW", "AXP", "BLK", "SPG", "PLD", "AMT", "DHR", "ZTS", "NOW", "UBER"
]

# Configurazione Pagina
st.set_page_config(page_title="NQL - AI Finance V3", layout="wide")

st.title("🧠 NQL: Neural Quant Lab")

tab1, tab2 = st.tabs(["🚀 Laboratorio", "📘 Guida"])

# --- TAB 1: LABORATORIO ---
with tab1:
    col_input, col_main = st.columns([1, 4])

    with col_input:
        st.subheader("Impostazioni")
        
        # SELEZIONE MERCATO E TICKER
        market_choice = st.selectbox("1. Seleziona Mercato", ["NASDAQ", "NYSE"])
        
        if market_choice == "NASDAQ":
            ticker_list = sorted(NASDAQ_TOP)
        else:
            ticker_list = sorted(NYSE_TOP)
            
        ticker = st.selectbox("2. Cerca Titolo", ticker_list, help="Scrivi per cercare...")

        st.markdown("---")
        start_date = st.date_input("Data Inizio", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("Data Fine", value=pd.to_datetime("today"))
        
        st.markdown("---")
        st.markdown("**Parametri AI**")
        look_back = st.slider("Memoria (Giorni)", 10, 90, 60)
        epochs = st.slider("Epoche", 1, 30, 10)
        neurons = st.slider("Neuroni", 20, 100, 50)
        
        run_btn = st.button("Avvia Analisi", type="primary")

    with col_main:
        if run_btn:
            # Container per lo stato (evita sfarfallii)
            status_box = st.status("Avvio sistema...", expanded=True)
            
            try:
                # 1. DOWNLOAD
                status_box.write(f"📥 Scaricamento dati storici: {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Gestione MultiIndex (fix per yfinance recente)
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        data = data.xs(ticker, axis=1, level=1)
                    except KeyError:
                        # Fallback se la struttura è diversa
                        pass

                if data.empty or len(data) < look_back + 20:
                    status_box.update(label="Errore Dati", state="error")
                    st.error(f"Dati insufficienti per {ticker}. Prova ad ampliare il periodo o cambiare titolo.")
                else:
                    # Selezioniamo solo la colonna Close
                    if 'Close' in data.columns:
                        df_close = data[['Close']]
                    else:
                        df_close = data.iloc[:, 0:1] # Prendi la prima colonna se non trovi 'Close'
                    
                    # Grafico preliminare
                    st.subheader(f"Andamento Storico: {ticker}")
                    st.line_chart(df_close)

                    # 2. PREPARAZIONE
                    status_box.write("⚙️ Normalizzazione e creazione sequenze...")
                    dataset = df_close.values.astype('float32')
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(dataset)

                    # Split Train/Test (80/20)
                    training_data_len = int(len(dataset) * 0.8)
                    train_data = scaled_data[0:training_data_len, :]

                    # Funzione creazione dataset
                    def create_sequences(dataset, look_back):
                        X, Y = [], []
                        for i in range(len(dataset) - look_back - 1):
                            X.append(dataset[i:(i + look_back), 0])
                            Y.append(dataset[i + look_back, 0])
                        return np.array(X), np.array(Y)

                    x_train, y_train = create_sequences(train_data, look_back)
                    
                    # Reshape per LSTM [samples, time steps, features]
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                    # 3. TRAINING
                    status_box.write(f"🧠 Addestramento Rete Neurale ({epochs} epoche)...")
                    
                    model = Sequential()
                    model.add(LSTM(neurons, return_sequences=True, input_shape=(look_back, 1)))
                    model.add(LSTM(neurons, return_sequences=False))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')

                    # Custom Callback per Streamlit
                    progress_bar = status_box.empty()
                    prog_bar_widget = st.progress(0)
                    
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            prog = (epoch + 1) / epochs
                            prog_bar_widget.progress(prog)
                    
                    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0, callbacks=[StreamlitCallback()])
                    
                    status_box.write("✅ Training completato. Elaborazione previsioni...")

                    # 4. TESTING
                    test_data = scaled_data[training_data_len - look_back:, :]
                    x_test, y_test = create_sequences(test_data, look_back)
                    
                    # I dati reali (non scalati) per il confronto
                    # Nota: y_test qui sopra è scalato, usiamo il dataset originale per il confronto reale
                    y_real = dataset[training_data_len + 1 : len(dataset) - look_back + look_back] 
                    # Correzione indice per allineamento
                    y_real = dataset[training_data_len:]
                    # Ricostruiamo la lunghezza corretta basata su x_test
                    y_real = dataset[len(dataset) - len(y_test):]

                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                    
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)

                    # 5. VISUALIZZAZIONE
                    status_box.update(label="Analisi Completata", state="complete", expanded=False)
                    
                    # Creiamo il dataframe per il grafico finale
                    train = df_close[:training_data_len]
                    valid = df_close[len(df_close) - len(predictions):].copy()
                    valid['Predictions'] = predictions

                    st.markdown("### 🔮 Risultati Previsione")
                    
                    # Grafico Matplotlib
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.set_title(f'Modello Neurale su {ticker}')
                    ax.set_xlabel('Data')
                    ax.set_ylabel('Prezzo ($)')
                    ax.plot(train['Close'], label='Dati Training')
                    ax.plot(valid['Close'], label='Dati Reali (Test)')
                    ax.plot(valid['Predictions'], label='Previsione AI', color='red')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig) # Importante: libera la memoria!

            except Exception as e:
                status_box.update(label="Errore Critico", state="error")
                st.error(f"Si è verificato un errore imprevisto: {e}")

# --- TAB 2: GUIDA ---
with tab2:
    st.header("Come usare la Ricerca")
    st.markdown("""
    1.  **Seleziona Mercato:** Scegli tra NASDAQ (Tech) o NYSE (Classici).
    2.  **Cerca Titolo:** Clicca sulla casella e inizia a scrivere. Es. scrivi "APP" e vedrai comparire "AAPL".
    3.  **Avvia:** Il sistema scaricherà i dati freschi.
    
    **Nota:** La lista dei titoli include le 50 aziende più grandi per ogni mercato per garantire stabilità e velocità.
    """)