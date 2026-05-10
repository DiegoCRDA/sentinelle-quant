import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
st.set_page_config(page_title="Sentinelle v2.9.1 - Macro Focus", layout="centered")
st.title("🛡️ Sentinelle Quant Pro v2.9.1")

exchange = ccxt.kraken()

# --- SYMBOLES ---
# Note : DXY est maintenant inclus par défaut
default_symbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'RENDER/USD', 'FET/USD', 
    'STX/USD', 'DOT/USD', 'NEAR/USD', 'SUI/USD', 'LINK/USD', 
    'PAXG/USD', 'GOOGL/USD', 'DXY'
]

st.sidebar.header("⚙️ Configuration")
selected_symbols = st.sidebar.multiselect("Matrice d'actifs :", options=default_symbols, default=default_symbols)
focus_symbol = st.sidebar.selectbox("Focus Tactique :", options=selected_symbols, index=selected_symbols.index('NEAR/USD') if 'NEAR/USD' in selected_symbols else 0)
atr_mult = st.sidebar.slider("Sensibilité Stop (ATR) :", 1.0, 3.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.header("📋 Ma Position Live")
entry_price = st.sidebar.number_input("Prix d'entrée ($)", value=0.0, step=0.0001, format="%.4f")
leverage = st.sidebar.number_input("Levier (x)", value=1, min_value=1)

if st.button("⚡ Rafraîchir tout le système"):
    st.cache_data.clear()

# --- MOTEUR HYBRIDE ---
def fetch_parallel(symbol):
    try:
        if symbol == "GOOGL/USD" or symbol == "DXY":
            ticker_str = "GOOGL" if "GOOGL" in symbol else "DX-Y.NYB"
            yf_obj = yf.Ticker(ticker_str)
            df = yf_obj.history(period="2d", interval="15m")
            df4h = yf_obj.history(period="1mo", interval="1h")
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
    d15, v15, d4h, prices = {}, {}, {}, {}
    data_15m, data_4h = {}, {}
    
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

    s15 = np.linalg.eigh(pd.DataFrame(data_15m).dropna().pct_change().corr())[0][-1] if len(data_15m) > 5 else 0
    s4 = np.linalg.eigh(pd.DataFrame(data_4h).dropna().pct_change().corr())[0][-1] if len(data_4h) > 5 else 0
    
    # ATR Focus
    try:
        of = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=30)
        dff = pd.DataFrame(of, columns=['t','o','h','l','c','v'])
        dff['tr'] = np.maximum(dff['h']-dff['l'], np.maximum(abs(dff['h']-dff['c'].shift(1)), abs(dff['l']-dff['c'].shift(1))))
        atr_p = (dff['tr'].rolling(14).mean().iloc[-1] / prices[focus_s]) * 100
    except: atr_p = 0.0

    return s15, s4, d15, v15, prices.get(focus_s, 0), atr_p, prices.get('DXY', 0)

# --- EXÉCUTION & AFFICHAGE ---
with st.spinner("Calcul des flux macro..."):
    s15, s4, d15, v15, p_f, atr, p_dxy = update_all(selected_symbols, focus_symbol)

# Conseil d'Action
st.header("💡 Conseil d'Action Rationnel")
state = d15.get(focus_symbol)
if s4 > 7.0 and state == "BUY" and v15.get(focus_symbol, 0) > 1.0:
    st.success(f"🔥 ACHAT (HAUTE CONVICTION) - Entrée : {p_f * (1-(atr/100)):.4f} $")
else: st.info("🧘 OBSERVATION - Pas de confluence.")

# --- NOUVELLE SECTION MACRO ---
st.divider()
col_m1, col_m2 = st.columns(2)
col_m1.metric("💵 Index Dollar (DXY)", f"{p_dxy:.2f}")
dxy_state = d15.get('DXY', 'NEUTRAL')
if dxy_state == "BUY": col_m2.error("DXY Fort (Risque ⚠️)")
elif dxy_state == "SELL": col_m2.success("DXY Faible (Boost 🚀)")
else: col_m2.info("DXY Stable")

# Métriques Système
st.divider()
st.subheader("📊 Métriques Système")
m1, m2, m3 = st.columns(3)
m1.metric("Cohésion 4h", f"{s4:.2f}")
m2.metric("Volatilité", f"{atr:.2f}%")
m3.metric("Prix Focus", f"{p_f:.4f} $")

# Radar
st.subheader("📡 Radar de Force & Macro")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E7D32' if d15.get(s) == "BUY" else ('#C62828' if d15.get(s) == "SELL" else '#757575') for s in selected_symbols]
ax.barh(selected_symbols, [v15.get(s, 0) for s in selected_symbols], color=colors)
ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
st.pyplot(fig)
