import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page pour un affichage mobile optimal
st.set_page_config(page_title="Sentinelle Quant Pro", layout="centered")

st.title("🛡️ Sentinelle Quant Pro")
st.write("Votre terminal quantique de poche.")

# Initialisation de l'API Kraken
exchange = ccxt.kraken()

# --- ESPACE CONFIGURATION DEPUIS TON IPHONE ---
st.sidebar.header("⚙️ Configuration")

all_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD', 'NEAR/USD', 'RENDER/USD', 'STX/USD', 'SUI/USD', 'FET/USD']

# Menu multi-sélection des jetons
selected_symbols = st.sidebar.multiselect(
    "Actifs de la Matrice :",
    options=all_symbols,
    default=all_symbols
)

# Choix de l'actif Focus pour l'ATR
default_focus = 'NEAR/USD' if 'NEAR/USD' in selected_symbols else selected_symbols[0]
focus_symbol = st.sidebar.selectbox(
    "Sélectionne le Jeton FOCUS (ATR) :",
    options=selected_symbols,
    index=selected_symbols.index(default_focus)
)

timeframe = st.sidebar.selectbox("Timeframe de calcul :", ['15m', '1h', '4h'], index=0)
atr_multiplier = st.sidebar.slider("Multiplicateur ATR (Stop-Loss) :", 1.0, 3.0, 2.0, 0.1)

# Bouton de rafraîchissement manuel
if st.button("🔄 Actualiser les données"):
    st.cache_data.clear()

# --- CACHE DES CALCULS POUR ÉVITER LES BAN API ---
@st.cache_data(ttl=15)
def fetch_and_calculate(symbols, tf, focus_s):
    data_dict = {}
    vols_dict = {}
    delta_dict = {}
    prices_dict = {}
    
    # 1. Récupération des données Kraken
    for s in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(s, timeframe=tf, limit=60)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df_closed = df.iloc[:-1].copy()
            data_dict[s] = df_closed['c']
            prices_dict[s] = df_closed['c'].iloc[-1]
            
            # RVOL
            recent_vol = df_closed['v'].iloc[-3:].mean()
            avg_vol = df_closed['v'].rolling(window=30).mean().iloc[-1]
            vols_dict[s] = recent_vol / avg_vol if avg_vol > 0 else 0
            
            # Delta CVD
            df_closed['single_delta'] = np.where(df_closed['c'] > df_closed['o'], df_closed['v'], -df_closed['v'])
            smoothed_delta = df_closed['single_delta'].ewm(span=5, adjust=False).mean().iloc[-1]
            
            if smoothed_delta > (recent_vol * 0.1):
                delta_dict[s] = "BUY"
            elif smoothed_delta < -(recent_vol * 0.1):
                delta_dict[s] = "SELL"
            else:
                delta_dict[s] = "NEUTRAL"
        except:
            continue
            
    # 2. Calcul du Signal de Cohésion de la Matrice (Valeurs propres)
    df_close = pd.DataFrame(data_dict).dropna()
    weights_dict = {s: 0.0 for s in symbols}
    signal_mtf = 0.0
    
    if len(df_close) > 5:
        returns = np.log(df_close / df_close.shift(1)).dropna()
        vals, vecs = np.linalg.eigh(returns.corr())
        signal_mtf = vals[-1]
        for idx, col in enumerate(df_close.columns):
            weights_dict[col] = np.abs(vecs[idx, -1])
            
    # 3. Calcul ATR pour le Focus
    try:
        ohlcv_focus = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=20)
        df_focus = pd.DataFrame(ohlcv_focus, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_focus_closed = df_focus.iloc[:-1].copy()
        df_focus_closed['tr'] = np.maximum(df_focus_closed['h'] - df_focus_closed['l'], 
                                         np.maximum(abs(df_focus_closed['h'] - df_focus_closed['c'].shift(1)), 
                                                    abs(df_focus_closed['l'] - df_focus_closed['c'].shift(1))))
        atr_pct = (df_focus_closed['tr'].rolling(14).mean().iloc[-1] / df_focus_closed['c'].iloc[-1]) * 100
        price_focus = df_focus_closed['c'].iloc[-1]
    except:
        atr_pct, price_focus = 0.0, 0.0
        
    # 4. Données Macro
    try:
        global_assets = {'BTC-USD': 'BTC', '^IXIC': 'Nasdaq'}
        m_data = yf.download(list(global_assets.keys()), period="1mo", progress=False)['Close'].dropna()
        m_corr = np.log(m_data / m_data.shift(1)).dropna().corr()
        corr_val = m_corr.loc['BTC-USD', '^IXIC']
    except:
        corr_val = 0.0
        
    return signal_mtf, weights_dict, vols_dict, delta_dict, price_focus, atr_pct, corr_val

# Exécution des calculs
with st.spinner("Analyse quantitative en cours..."):
    sig_mtf, weights, rvols, deltas, price_focus, atr_pct, corr_val = fetch_and_calculate(selected_symbols, timeframe, focus_symbol)

# --- AFFICHAGE DES CLÉS ---
st.header(f"🎯 Focus : {focus_symbol}")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(label="Prix Clôture", value=f"{price_focus:.4f} $")
with c2:
    stop_ideal = price_focus * (1 - (atr_multiplier * atr_pct / 100))
    st.metric(label=f"Stop Idéal ({atr_multiplier}x ATR)", value=f"{stop_ideal:.4f} $")
with c3:
    st.metric(label="Volatilité (ATR 1h)", value=f"{atr_pct:.2f}%")

st.subheader("📊 Métriques Système")
col_a, col_b = st.columns(2)
with col_a:
    st.metric(label=f"Cohésion Matrice ({timeframe})", value=f"{sig_mtf:.2f}")
with col_b:
    st.metric(label="Corrélation Bourse (Nasdaq)", value=f"{corr_val:.2f}")

# --- GRAPHICS COMPATIBLES STREAMLIT ---
st.subheader("📈 Graphiques de Force")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Graphique de Gauche : Signal MTF (Weights)
weights_y = [weights.get(s, 0) for s in selected_symbols]
ax1.barh(selected_symbols, weights_y, color=['#FF5722' if focus_symbol in s else '#3399FF' for s in selected_symbols])
ax1.set_title(f"SIGNAL MTF (Cohésion : {sig_mtf:.2f})")

# Graphique de Droite : RVOL & Delta
colors_rvol = []
for s in selected_symbols:
    d = deltas.get(s, "NEUTRAL")
    if d == "BUY":
        colors_rvol.append('#4CAF50')
    elif d == "SELL":
        colors_rvol.append('#F44336')
    else:
        colors_rvol.append('#9E9E9E')

rvols_y = [rvols.get(s, 0) for s in selected_symbols]
ax2.barh(selected_symbols, rvols_y, color=colors_rvol)
ax2.axvline(1.0, color='black', linestyle='--')
ax2.set_title("VOLUME RELATIF & AGRESSIVITÉ PRO")

st.pyplot(fig)
