import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Configuration de la page
st.set_page_config(page_title="Sentinelle v2.7 - Rational Engine", layout="centered")
st.title("🛡️ Sentinelle Quant Pro v2.7")

exchange = ccxt.kraken()

# --- CONFIG ---
default_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'RENDER/USD', 'FET/USD', 'STX/USD', 'DOT/USD', 'NEAR/USD', 'SUI/USD', 'LINK/USD', 'PAXG/USD', 'GOOGL/USD']
st.sidebar.header("⚙️ Paramètres")
selected_symbols = st.sidebar.multiselect("Matrice :", options=default_symbols, default=default_symbols)
focus_symbol = st.sidebar.selectbox("Focus :", options=selected_symbols, index=selected_symbols.index('DOT/USD') if 'DOT/USD' in selected_symbols else 0)
atr_mult = st.sidebar.slider("Multiplicateur Stop (ATR) :", 1.0, 3.0, 2.0, 0.1)

if st.button("⚡ Rafraîchir le Système de Décision"):
    st.cache_data.clear()

def fetch_parallel(symbol):
    try:
        o15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        o4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
        ticker = exchange.fetch_ticker(symbol)
        return symbol, o15, o4, ticker['last']
    except: return symbol, None, None, None

@st.cache_data(ttl=15)
def update_all(symbols, focus_s):
    data_15m, vols_15m, delta_15m = {}, {}, {}
    data_4h, prices = {}, {}
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_parallel, symbols))

    for sym, o15, o4, p in results:
        if o15 and o4:
            prices[sym] = p
            df15 = pd.DataFrame(o15, columns=['t','o','h','l','c','v'])
            data_15m[sym] = df15['c']
            
            # Calcul Stabilité Delta (Loi de Wigner / Filtrage du bruit)
            df15['v_delta'] = np.where(df15['c'] > df15['o'], df15['v'], -df15['v'])
            smoothed = df15['v_delta'].ewm(span=8, adjust=False).mean().iloc[-1]
            threshold = df15['v'].rolling(20).mean().iloc[-1] * 0.10
            delta_15m[sym] = "BUY" if smoothed > threshold else ("SELL" if smoothed < -threshold else "NEUTRAL")
            
            # RVOL (Volume Relatif)
            vols_15m[sym] = df15['v'].iloc[-5:].mean() / df15['v'].rolling(30).mean().iloc[-1]
            
            df4 = pd.DataFrame(o4, columns=['t','o','h','l','c','v'])
            data_4h[sym] = df4['c']

    # Métriques Système (Matrices de Corrélation)
    df_c15 = pd.DataFrame(data_15m).dropna().pct_change().dropna()
    sig15 = np.linalg.eigh(df_c15.corr())[0][-1] if not df_c15.empty else 0
    df_c4 = pd.DataFrame(data_4h).dropna().pct_change().dropna()
    sig4 = np.linalg.eigh(df_c4.corr())[0][-1] if not df_c4.empty else 0
    
    # Nasdaq Corr
    try:
        m_data = yf.download(['BTC-USD', '^IXIC'], period="1mo", progress=False)['Close'].dropna()
        corr_val = np.log(m_data / m_data.shift(1)).dropna().corr().loc['BTC-USD', '^IXIC']
    except: corr_val = 0.0

    # ATR Conseil
    try:
        of = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=30)
        dff = pd.DataFrame(of, columns=['t','o','h','l','c','v'])
        dff['tr'] = np.maximum(dff['h']-dff['l'], np.maximum(abs(dff['h']-dff['c'].shift(1)), abs(dff['l']-dff['c'].shift(1))))
        atr_val = dff['tr'].rolling(14).mean().iloc[-1]
        atr_pct = (atr_val / prices[focus_s]) * 100
    except: atr_pct = 0.0

    return sig15, sig4, corr_val, delta_15m, vols_15m, prices.get(focus_s, 0), atr_pct

# --- MOTEUR DE DÉCISION RATIONNELLE ---
with st.spinner("Analyse des flux en cascade..."):
    s15, s4, nas, d15, v15, p_f, atr = update_all(selected_symbols, focus_symbol)

st.header("💡 Conseil d'Action Rationnel")

state_15m = d15.get(focus_symbol)
rvol_focus = v15.get(focus_symbol, 0)
btc_state = d15.get('BTC/USD', 'NEUTRAL')

# Conditions logiques
volume_valide = rvol_focus > 1.0
tendance_macro = s4 > 7.0
btc_favorable = btc_state != "SELL"

if tendance_macro:
    if state_15m == "BUY":
        if btc_favorable and volume_valide:
            st.success(f"🔥 ACHAT (HAUTE CONVICTION) - Entrée : {p_f * (1-(atr/100)):.4f} $")
            st.write(f"Confluence détectée : RVOL ({rvol_focus:.2f}) > 1.0 et BTC stable/haussier.")
        elif not volume_valide:
            st.warning(f"⚠️ ATTENTE - Signal haussier sans volume (RVOL: {rvol_focus:.2f}).")
        else:
            st.info("✋ NEUTRE - Signal haussier sur l'actif mais BTC est en pression vendeuse.")
            
    elif state_15m == "SELL":
        if btc_state == "SELL" and volume_valide:
            st.error(f"📉 VENTE (SHORT) - Entrée : {p_f * (1+(atr/100)):.4f} $")
        else:
            st.info("✋ NEUTRE - Pression vendeuse détectée mais non confirmée par le flux global.")
    else:
        st.info("🧘 OBSERVATION - Pas de flux directionnel lissé (Bruit dominant).")
else:
    st.info("💤 MARCHÉ ENDORMI - Faible cohésion macro, privilégier le cash.")

# --- AFFICHAGE MÉTRIQUES ---
st.subheader("📊 Métriques Système")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cohésion 15m", f"{s15:.2f}")
m2.metric("Cohésion 4h", f"{s4:.2f}")
m3.metric("Corr. Nasdaq", f"{nas:.2f}")
m4.metric("Volatilité", f"{atr:.2f}%")

st.write(f"**Prix actuel {focus_symbol}:** {p_f:.4f} $")

# Radar Graphique
st.subheader("📡 Radar de Force & Stabilité")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E7D32' if d15.get(s) == "BUY" else ('#C62828' if d15.get(s) == "SELL" else '#757575') for s in selected_symbols]
ax.barh(selected_symbols, [v15.get(s, 0) for s in selected_symbols], color=colors)
ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
st.pyplot(fig)
