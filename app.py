import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
st.set_page_config(page_title="Sentinelle v3.2 - Data Integrity", layout="centered")
st.title("🛡️ Sentinelle Quant Pro v3.2")

exchange = ccxt.kraken()

# --- SYMBOLES ---
default_symbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'RENDER/USD', 'FET/USD', 
    'STX/USD', 'DOT/USD', 'NEAR/USD', 'SUI/USD', 'LINK/USD', 
    'PAXG/USD', 'GOOGL/USD', 'DXY'
]

st.sidebar.header("⚙️ Matrice")
selected_symbols = st.sidebar.multiselect("Actifs :", options=default_symbols, default=default_symbols)
focus_symbol = st.sidebar.selectbox("Actif Focus :", options=selected_symbols, index=0)
atr_mult = st.sidebar.slider("Sensibilité Stop (ATR) :", 1.0, 3.0, 2.0, 0.1)

if st.button("⚡ Actualiser les flux"):
    st.cache_data.clear()

# --- MOTEUR HYBRIDE ---
def fetch_parallel(symbol):
    try:
        if symbol == "GOOGL/USD" or symbol == "DXY":
            yf_ticker = "DX-Y.NYB" if symbol == "DXY" else "GOOGL"
            ticker_yf = yf.Ticker(yf_ticker)
            df = ticker_yf.history(period="2d", interval="15m")
            df4h = ticker_yf.history(period="1mo", interval="1h")
            o15 = [[int(t.timestamp()*1000), r.Open, r.High, r.Low, r.Close, r.Volume] for t, r in df.iterrows()]
            o4 = [[int(t.timestamp()*1000), r.Open, r.High, r.Low, r.Close, r.Volume] for t, r in df4h.iterrows()]
            return symbol, o15, o4, df['Close'].iloc[-1]
        else:
            o15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
            o4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
            p = exchange.fetch_ticker(symbol)['last']
            return symbol, o15, o4, p
    except: return symbol, None, None, None

@st.cache_data(ttl=15)
def update_all(symbols, focus_s):
    d15, v15, prices, data_15m, data_4h = {}, {}, {}, {}, {}
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_parallel, symbols))

    for sym, o15, o4, p in results:
        if o15 and o4:
            prices[sym] = p
            df15 = pd.DataFrame(o15, columns=['t','o','h','l','c','v'])
            data_15m[sym] = df15['c']
            df15['v_delta'] = np.where(df15['c'] > df15['o'], df15['v'], -df15['v'])
            smoothed = df15['v_delta'].ewm(span=8, adjust=False).mean().iloc[-1]
            t = df15['v'].rolling(20).mean().iloc[-1] * 0.10
            d15[sym] = "BUY" if smoothed > t else ("SELL" if smoothed < -t else "NEUTRAL")
            v15[sym] = df15['v'].iloc[-5:].mean() / df15['v'].rolling(30).mean().iloc[-1]
            data_4h[sym] = pd.DataFrame(o4)[4]

    # Calcul des deux cohésions (Matrice de corrélation de Wigner)
    s15 = np.linalg.eigh(pd.DataFrame(data_15m).dropna().pct_change().dropna().corr())[0][-1] if len(data_15m) > 5 else 0
    s4 = np.linalg.eigh(pd.DataFrame(data_4h).dropna().pct_change().dropna().corr())[0][-1] if len(data_4h) > 5 else 0
    
    try:
        of = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=30)
        dff = pd.DataFrame(of, columns=['t','o','h','l','c','v'])
        dff['tr'] = np.maximum(dff['h']-dff['l'], np.maximum(abs(dff['h']-dff['c'].shift(1)), abs(dff['l']-dff['c'].shift(1))))
        atr_p = (dff['tr'].rolling(14).mean().iloc[-1] / prices[focus_s]) * 100
    except: atr_p = 0.0

    return s15, s4, d15, v15, prices.get(focus_s, 0), atr_p, prices.get('DXY', 0)

# --- AFFICHAGE ---
s15, s4, d15, v15, p_f, atr, p_dxy = update_all(selected_symbols, focus_symbol)

# Bloc Macro Dollar
st.metric("💵 Index Dollar (DXY)", f"{p_dxy:.2f}")
st.divider()

# Métriques Système et Actif Focus
st.subheader(f"📊 Analyse : {focus_symbol}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cohésion 15m", f"{s15:.2f}")
m2.metric("Cohésion 4h", f"{s4:.2f}")
m3.metric("Volatilité (1h)", f"{atr:.2f}%")
m4.metric(f"Prix {focus_symbol}", f"{p_f:.2f} $")

# Radar de Force
st.divider()
st.subheader("📡 Radar de Force (Volume & Direction)")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E7D32' if d15.get(s) == "BUY" else ('#C62828' if d15.get(s) == "SELL" else '#757575') for s in selected_symbols]
ax.barh(selected_symbols, [v15.get(s, 0) for s in selected_symbols], color=colors)
ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
st.pyplot(fig)
