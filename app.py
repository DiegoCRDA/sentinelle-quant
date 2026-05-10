import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Sentinelle v2.9 - Macro Master", layout="centered")
st.title("🛡️ Sentinelle Quant Pro v2.9")

exchange = ccxt.kraken()

# --- CONFIGURATION DES SYMBOLES ---
# Ajout de DXY pour le Dollar et GOOGL/USD
default_symbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'RENDER/USD', 'FET/USD', 
    'STX/USD', 'DOT/USD', 'NEAR/USD', 'SUI/USD', 'LINK/USD', 
    'PAXG/USD', 'GOOGL/USD', 'DXY'
]

# --- SIDEBAR ---
st.sidebar.header("⚙️ Paramètres Système")
selected_symbols = st.sidebar.multiselect("Matrice d'actifs :", options=default_symbols, default=default_symbols)
focus_symbol = st.sidebar.selectbox("Focus Tactique :", options=selected_symbols, index=selected_symbols.index('NEAR/USD') if 'NEAR/USD' in selected_symbols else 0)
atr_mult = st.sidebar.slider("Sensibilité Stop (ATR) :", 1.0, 3.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.header("📋 Ma Position Live")
entry_price = st.sidebar.number_input("Prix d'entrée ($)", value=0.0, step=0.0001, format="%.4f")
leverage = st.sidebar.number_input("Levier (x)", value=1, min_value=1, step=1)

if st.button("⚡ Rafraîchir le Système Hybride"):
    st.cache_data.clear()

# --- MOTEUR DE RÉCUPÉRATION HYBRIDE (Kraken + yfinance) ---
def fetch_parallel(symbol):
    try:
        # Cas spécifiques pour les actifs boursiers et macro (yfinance)
        if symbol in ["GOOGL/USD", "DXY"]:
            yf_ticker = "GOOGL" if "GOOGL" in symbol else "DX-Y.NYB"
            ticker_yf = yf.Ticker(yf_ticker)
            
            # Données 15m et 1h (pour simuler le 4h macro)
            df_yf = ticker_yf.history(period="2d", interval="15m")
            df_yf_4h = ticker_yf.history(period="1mo", interval="1h")
            
            o15 = [[int(t.timestamp()*1000), r.Open, r.High, r.Low, r.Close, r.Volume] for t, r in df_yf.iterrows()]
            o4 = [[int(t.timestamp()*1000), r.Open, r.High, r.Low, r.Close, r.Volume] for t, r in df_yf_4h.iterrows()]
            price = df_yf['Close'].iloc[-1]
            return symbol, o15, o4, price
        
        # Cas standard (Kraken Crypto)
        else:
            o15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
            o4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
            ticker = exchange.fetch_ticker(symbol)
            return symbol, o15, o4, ticker['last']
    except:
        return symbol, None, None, None

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
            
            # Filtrage du bruit (Wigner / EMA 8)
            df15['v_delta'] = np.where(df15['c'] > df15['o'], df15['v'], -df15['v'])
            smoothed = df15['v_delta'].ewm(span=8, adjust=False).mean().iloc[-1]
            threshold = df15['v'].rolling(20).mean().iloc[-1] * 0.10
            delta_15m[sym] = "BUY" if smoothed > threshold else ("SELL" if smoothed < -threshold else "NEUTRAL")
            vols_15m[sym] = df15['v'].iloc[-5:].mean() / df15['v'].rolling(30).mean().iloc[-1]
            
            df4 = pd.DataFrame(o4, columns=['t','o','h','l','c','v'])
            data_4h[sym] = df4['c']

    # Cohésions
    df_c15 = pd.DataFrame(data_15m).dropna().pct_change().dropna()
    sig15 = np.linalg.eigh(df_c15.corr())[0][-1] if not df_c15.empty else 0
    df_c4 = pd.DataFrame(data_4h).dropna().pct_change().dropna()
    sig4 = np.linalg.eigh(df_c4.corr())[0][-1] if not df_c4.empty else 0
    
    # Nasdaq Corr
    try:
        m_data = yf.download(['BTC-USD', '^IXIC'], period="1mo", progress=False)['Close'].dropna()
        corr_val = np.log(m_data / m_data.shift(1)).dropna().corr().loc['BTC-USD', '^IXIC']
    except: corr_val = 0.0

    # ATR Focus
    try:
        of = exchange.fetch_ohlcv(focus_s, timeframe='1h', limit=30)
        dff = pd.DataFrame(of, columns=['t','o','h','l','c','v'])
        dff['tr'] = np.maximum(dff['h']-dff['l'], np.maximum(abs(dff['h']-dff['c'].shift(1)), abs(dff['l']-dff['c'].shift(1))))
        atr_pct = (dff['tr'].rolling(14).mean().iloc[-1] / prices[focus_s]) * 100
    except: atr_pct = 0.0

    return sig15, sig4, corr_val, delta_15m, vols_15m, prices.get(focus_s, 0), atr_pct

# --- UI ---
with st.spinner("Analyse Macro-Crypto Hybride..."):
    s15, s4, nas, d15, v15, p_f, atr = update_all(selected_symbols, focus_symbol)

# Conseil Rationnel
st.header("💡 Conseil d'Action Rationnel")
state_15m = d15.get(focus_symbol)
rvol_focus = v15.get(focus_symbol, 0)
btc_state = d15.get('BTC/USD', 'NEUTRAL')

if s4 > 7.0:
    if state_15m == "BUY" and btc_state != "SELL" and rvol_focus > 1.0:
        st.success(f"🔥 ACHAT (HAUTE CONVICTION) - Cible entrée : {p_f * (1-(atr/100)):.4f} $")
    elif state_15m == "BUY" and rvol_focus <= 1.0:
        st.warning("⚠️ ATTENTE - Signal haussier sans volume réel.")
    else: st.info("🧘 OBSERVATION - Pas de confluence claire.")
else: st.info("💤 MARCHÉ ENDORMI - Faible cohésion macro.")

# Suivi Trade
if entry_price > 0:
    st.divider()
    st.header("🎯 Mon Trade en Cours")
    diff_pct = ((p_f - entry_price) / entry_price) * 100
    pnl_lev = diff_pct * leverage
    current_stop = p_f * (1 - (atr * atr_mult / 100))
    c1, c2 = st.columns(2)
    c1.metric("PnL Latent", f"{pnl_lev:.2f}%", delta=f"{diff_pct:.2f}% net")
    c2.metric("Stop Conseillé", f"{max(current_stop, entry_price):.4f} $", "Trailing" if current_stop > entry_price else "Break-even")

# Métriques Système
st.divider()
st.subheader("📊 Métriques Système")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cohésion 15m", f"{s15:.2f}")
m2.metric("Cohésion 4h", f"{s4:.2f}")
m3.metric("Corr. Nasdaq", f"{nas:.2f}")
m4.metric("Volatilité", f"{atr:.2f}%")

# Radar
st.subheader("📡 Radar de Force & Macro")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E7D32' if d15.get(s) == "BUY" else ('#C62828' if d15.get(s) == "SELL" else '#757575') for s in selected_symbols]
ax.barh(selected_symbols, [v15.get(s, 0) for s in selected_symbols], color=colors)
ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
st.pyplot(fig)
