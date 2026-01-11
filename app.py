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

# Creiamo le schede per separare l'azione dalla teoria
tab1, tab2 = st.tabs(["🚀 Laboratorio", "📘 Guida & Tutorial"])

# --- CONTENUTO SCHEDA 1: IL SOFTWARE ---
with tab1:
    st.markdown("### Pannello di Controllo")
    
    col1, col2 = st.columns([1, 3])

    with col1:
        st.info("Imposta qui i parametri della tua simulazione.")
        # --- INPUT ---
        ticker = st.text_input("Simbolo (Ticker)", "AAPL", help="Es: AAPL per Apple, NVDA per Nvidia, BTC-USD per Bitcoin")
        start_date = st.date_input("Data Inizio", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("Data Fine", value=pd.to_datetime("today"))
        
        st.markdown("---")
        st.markdown("**Parametri AI**")
        look_back = st.slider("Memoria (Giorni)", 10, 120, 60, help="Quanti giorni passati guardare per prevedere il domani.")
        epochs = st.slider("Epoche (Ripetizioni)", 1, 50, 10, help="Quante volte la rete studia l'intero set di dati.")
        neurons = st.slider("Neuroni (Potenza)", 10, 200, 50, help="Numero di unità di calcolo nella rete.")
        
        run_btn = st.button("Avvia Elaborazione NQL", type="primary")

    with col2:
        # --- LOGICA APPLICATIVA ---
        if run_btn:
            status_container = st.container()
            
            # Funzioni interne
            def load_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end)
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs(ticker, axis=1, level=1)
                return data

            def create_dataset(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            # Esecuzione
            with status_container:
                st.write(f"🔄 **Fase 1:** Scaricamento dati per {ticker}...")
                df = load_data(ticker, start_date, end_date)
            
            if df.empty:
                st.error("Nessun dato trovato! Controlla il simbolo (es. prova con AAPL).")
            else:
                # Pulizia dati
                if 'Close' in df.columns:
                    close_data = df[['Close']]
                else:
                    close_data = df.iloc[:, 0:1]

                # Pre-processing
                dataset = close_data.values
                training_data_len = int(np.ceil(len(dataset) * .80))
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(dataset)

                train_data = scaled_data[0:int(training_data_len), :]
                x_train, y_train = create_dataset(train_data, look_back)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                # Training
                with status_container:
                    st.write(f"🧠 **Fase 2:** Addestramento Rete Neurale in corso ({epochs} epoche)...")
                    st.write("Il processore sta cercando pattern matematici nei prezzi...")
                    my_bar = st.progress(0)

                model = Sequential()
                model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(neurons, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Callback per la barra di progresso
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        my_bar.progress((epoch + 1) / epochs)
                
                model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=[StreamlitCallback()], verbose=0)
                
                # Test
                with status_container:
                    st.success("✅ Addestramento Completato! Generazione grafico...")
                
                test_data = scaled_data[training_data_len - look_back: , :]
                x_test, y_test = create_dataset(test_data, look_back)
                y_test = dataset[training_data_len:, :]
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                
                # Calcolo Errore
                rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
                
                # Grafico
                valid = close_data[training_data_len:].copy()
                valid['Predictions'] = predictions
                
                st.metric("Errore Medio (RMSE)", f"{rmse:.2f}", help="Più è basso, meglio è.")
                
                fig, ax = plt.subplots(figsize=(14,7))
                ax.set_title(f"Previsione AI su {ticker}")
                ax.set_xlabel("Data")
                ax.set_ylabel("Prezzo ($)")
                ax.plot(close_data[:training_data_len]['Close'], label='Dati Training (Il passato studiato)', color='blue')
                ax.plot(valid['Close'], label='Dati Reali (Cosa è successo davvero)', color='green')
                ax.plot(valid['Predictions'], label='Previsione AI (Cosa la rete ha ipotizzato)', color='red', linestyle='--')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.caption("Nota: La linea rossa tratteggiata è generata dalla rete neurale basandosi SOLO sui dati precedenti.")

# --- CONTENUTO SCHEDA 2: LA GUIDA ---
with tab2:
    st.header("📘 Manuale di NQL")
    st.markdown("""
    Benvenuto nel tuo laboratorio di Reti Neurali. Ecco come interpretare quello che vedi.

    ### 1. I Concetti Base
    Stiamo usando una rete **LSTM (Long Short-Term Memory)**. Immaginala come uno studente che legge il grafico di borsa.
    - **Training Set (Linea Blu):** Sono i libri di storia su cui lo studente ha studiato.
    - **Test Set (Linea Verde):** È l'esame finale. Dati che lo studente NON aveva mai visto prima.
    - **Previsione (Linea Rossa):** Le risposte che lo studente ha dato all'esame.

    ### 2. I Parametri (Sidebar)
    
    #### **Memoria (Lookback)**
    *   **Cosa fa:** Decide quanti giorni indietro guardare per capire oggi.
    *   *Esempio:* Se metti "60", per prevedere il prezzo di domani, la rete guarda i prezzi degli ultimi 2 mesi.
    *   *Consiglio:* Tra 30 e 90 è un buon range. Troppo poco perde il contesto, troppo rischia confusione.

    #### **Epoche (Epochs)**
    *   **Cosa fa:** Quante volte lo studente rilegge l'intero libro di storia.
    *   *Basso (1-5):* Studio superficiale. La rete potrebbe non capire i pattern complessi (Underfitting).
    *   *Alto (50+):* Studio "a memoria". La rete impara a memoria i dati passati ma fallisce su quelli nuovi (Overfitting).
    *   *Consiglio:* Inizia con 10 o 20.

    #### **Neuroni**
    *   **Cosa fa:** La capacità cerebrale della rete.
    *   *Consiglio:* 50 è un buon equilibrio per il tuo Mac Mini M4.

    ### 3. Interpretare il Grafico
    Il tuo obiettivo è che la **Linea Rossa (Previsione)** sia il più possibile sovrapposta alla **Linea Verde (Realtà)**.
    - Se la linea rossa è piatta o lontanissima: La rete non ha imparato (prova ad aumentare le epoche).
    - Se la linea rossa segue perfettamente quella verde ma in ritardo: La rete sta solo copiando il prezzo di ieri (è un classico "trucco" che le reti fanno se non configurate bene).
    """)