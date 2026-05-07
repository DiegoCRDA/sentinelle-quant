import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Config
st.set_page_config(page_title="Sentinelle v2.5 - Stability Pro", layout="centered")
st.title("🛡️ Sentinelle Quant Pro v2.5")
st.write("Moteur de stabilité par Lissage Exponentiel.")

exchange = ccxt.kraken()

# --- CONFIGURATION ---
default_symbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'RENDER/USD', 'FET/USD', 
    'STX/USD', 'DOT/USD', 'NEAR/USD', 'SUI/USD', 'LINK/USD', 
    'PAXG/USD', 'GOOGL/USD'
]

st.sidebar.header("⚙️ Paramètres")
selected_symbols = st.sidebar.multiselect("Matrice :", options=default_symbols, default=default_symbols)
focus_symbol = st.sidebar.selectbox("Focus :", options=selected_symbols, index=0)
atr_mult = st.sidebar.slider("Multiplicateur Stop (ATR) :", 1.0, 3.0, 2.0, 0.1)

if st.button("⚡ Rafraîchir (Moteur Parallèle)"):
    st.cache_data.clear()

def fetch_data_parallel(symbol):
    try:
        # On demande un peu plus de bougies pour le lissage (100)
        o15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        o4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
        ticker = exchange.fetch_ticker(symbol)
        return symbol, o15, o4, ticker['last']
    except:
        return symbol, None, None, None

@st.cache_data(ttl=15)
def update_matrix(symbols, focus_s):
    data_15m, vols_15m, delta_15m = {}, {}, {}
    data_4h, vols_4h, delta_4h = {}, {}, {}
    prices = {}

    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_data_parallel, symbols))

    for sym, o15, o4, p in results:
        if o15 and o4:
            prices[sym] = p
            
            # --- CALCUL STABILISÉ (15m) ---
            df15 = pd.DataFrame(o15, columns=['t','o','h','l','c','v'])
            data_15m[sym] = df15['c']
            
            # 1. Calcul du Delta Volume
            df15['v_delta'] = np.where(df15['c'] > df15['o'], df15['v'], -df15['v'])
            # 2. Lissage par EMA (période 8 pour la réactivité stable)
            smoothed_delta = df15['v_delta'].ewm(span=8, adjust=False).mean().iloc[-1]
            # 3. Zone Neutre (10% du volume moyen)
            threshold = df15['v'].rolling(20).mean().iloc[-1] * 0.10
            
            if smoothed_delta > threshold: delta_15m[sym] = "BUY"
            elif smoothed_delta < -threshold: delta_15m[sym] = "SELL"
            else: delta_15m[sym] = "NEUTRAL"
            
            vols_15m[sym] = df15['v'].iloc[-5:].mean() / df15['v'].rolling(30).mean().iloc[-1]

            # --- CALCUL MACRO (4h) ---
            df4 = pd.DataFrame(o4, columns=['t','o','h','l','c','v'])
            data_4h[sym] = df4['c']
            # Même logique de lissage pour la Macro
            v_delta_4h = np.where(df4['c'] > df4['o'], df4['v'], -df4['v'])
            smoothed_4h = pd.Series(v_delta_4h).ewm(span=8, adjust=False).mean().iloc[-1]
            t4h = df4['v'].rolling(20).mean().iloc[-1] * 0.10
            
            if smoothed_4h > t4h: delta_4h[sym] = "BUY"
            elif smoothed_4h < -t4h: delta_4h[sym] = "SELL"
            else: delta_4h[sym] = "NEUTRAL"

    # Cohésions (Matrix Correl)
    sig15 = np.linalg.eigh(pd.DataFrame(data_15m).pct_change().corr())[0][-1] if len(data_15m)>5 else 0
    sig4 = np.linalg.eigh(pd.DataFrame(data_4h).pct_change().corr())[0][-1] if len(data_4h)>5 else 0

    # ATR pour conseil
    try:
        of = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=30)
        dff = pd.DataFrame(of, columns=['t','o','h','l','c','v'])
        dff['tr'] = np.maximum(dff['h']-dff['l'], np.maximum(abs(dff['h']-dff['c'].shift(1)), abs(dff['l']-dff['c'].shift(1))))
        atr_val = dff['tr'].rolling(14).mean().iloc[-1]
        atr_pct = (atr_val / prices[focus_s]) * 100
    except: atr_pct = 0.0

    return sig15, sig4, delta_15m, vols_15m, prices.get(focus_s, 0), atr_pct, delta_4h

# --- LOGIQUE D'AFFICHAGE ---
with st.spinner("Analyse stable en cours..."):
    s15, s4, d15, v15, p_f, atr, d4 = update_matrix(selected_symbols, focus_symbol)

# Bloc Conseil
st.header("💡 Conseil d'Action")
current_state = d15.get(focus_symbol)
if current_state == "BUY" and s4 > 7.0:
    st.success(f"MODE : ACHAT (LONG) - Cible : {p_f * (1-(atr/100)):.4f} $")
elif current_state == "SELL" and s4 > 7.0:
    st.error(f"MODE : VENTE (SHORT) - Cible : {p_f * (1+(atr/100)):.4f} $")
else:
    st.info("MODE : OBSERVATION (Pas de direction claire)")

# Métriques
m1, m2, m3 = st.columns(3)
m1.metric("Prix Focus", f"{p_f:.4f} $")
m2.metric("Cohésion Macro", f"{s4:.2f}")
m3.metric("Volatilité", f"{atr:.2f}%")

# Graphique
st.subheader("📊 Radar de Force & Stabilité")
fig, ax = plt.subplots(figsize=(10, 6))
# On utilise les couleurs : Vert (BUY), Rouge (SELL), Gris (NEUTRAL)
colors = []
for s in selected_symbols:
    state = d15.get(s, "NEUTRAL")
    if state == "BUY": colors.append('#2E7D32') # Vert foncé
    elif state == "SELL": colors.append('#C62828') # Rouge foncé
    else: colors.append('#757575') # Gris

ax.barh(selected_symbols, [v15.get(s, 0) for s in selected_symbols], color=colors)
ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
ax.set_title("Volume Relatif & Direction Lissée (15m)")
st.pyplot(fig)
