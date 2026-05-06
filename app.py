import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page pour un affichage mobile optimal
st.set_page_config(page_title="Sentinelle Quant Pro v2", layout="centered")

st.title("🛡️ Sentinelle Quant Pro v2")
st.write("Votre terminal quantique de poche.")

# Initialisation de l'API Kraken
exchange = ccxt.kraken()

# --- ESPACE CONFIGURATION DEPUIS TON IPHONE ---
st.sidebar.header("⚙️ Configuration")

# Ajout de SUI, FET et PAXG (Pax Gold)
all_symbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD', 
    'LINK/USD', 'NEAR/USD', 'RENDER/USD', 'STX/USD', 'SUI/USD', 
    'FET/USD', 'PAXG/USD'
]

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

# Permet de basculer la vue des graphiques, tout en gardant le double calcul de cohésion à l'écran
timeframe_view = st.sidebar.selectbox(
    "Vue des Graphiques (Timeframe) :", 
    ['15m', '4h'], 
    index=0
)
atr_multiplier = st.sidebar.slider("Multiplicateur ATR (Stop-Loss) :", 1.0, 3.0, 2.0, 0.1)

# Bouton de rafraîchissement manuel
if st.button("🔄 Actualiser les données"):
    st.cache_data.clear()

# --- CACHE DES CALCULS POUR ÉVITER LES BAN API ---
@st.cache_data(ttl=15)
def fetch_and_calculate_dual(symbols, focus_s):
    # Structures de données 15m (Tactique)
    data_15m = {}
    vols_15m = {}
    delta_15m = {}
    
    # Structures de données 4h (Macro)
    data_4h = {}
    vols_4h = {}
    delta_4h = {}
    
    # 1. Récupération des données Kraken en une seule boucle
    for s in symbols:
        try:
            # --- TACTIQUE (15m) ---
            ohlcv_15m = exchange.fetch_ohlcv(s, timeframe='15m', limit=60)
            df_15m = pd.DataFrame(ohlcv_15m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df_15m_closed = df_15m.iloc[:-1].copy()
            data_15m[s] = df_15m_closed['c']
            
            # RVOL 15m
            recent_vol_15m = df_15m_closed['v'].iloc[-3:].mean()
            avg_vol_15m = df_15m_closed['v'].rolling(window=30).mean().iloc[-1]
            vols_15m[s] = recent_vol_15m / avg_vol_15m if avg_vol_15m > 0 else 0
            
            # Delta CVD 15m
            df_15m_closed['single_delta'] = np.where(df_15m_closed['c'] > df_15m_closed['o'], df_15m_closed['v'], -df_15m_closed['v'])
            smoothed_delta_15m = df_15m_closed['single_delta'].ewm(span=5, adjust=False).mean().iloc[-1]
            
            if smoothed_delta_15m > (recent_vol_15m * 0.1):
                delta_15m[s] = "BUY"
            elif smoothed_delta_15m < -(recent_vol_15m * 0.1):
                delta_15m[s] = "SELL"
            else:
                delta_15m[s] = "NEUTRAL"
                
            # --- MACRO (4h) ---
            ohlcv_4h = exchange.fetch_ohlcv(s, timeframe='4h', limit=60)
            df_4h = pd.DataFrame(ohlcv_4h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df_4h_closed = df_4h.iloc[:-1].copy()
            data_4h[s] = df_4h_closed['c']
            
            # RVOL 4h
            recent_vol_4h = df_4h_closed['v'].iloc[-3:].mean()
            avg_vol_4h = df_4h_closed['v'].rolling(window=30).mean().iloc[-1]
            vols_4h[s] = recent_vol_4h / avg_vol_4h if avg_vol_4h > 0 else 0
            
            # Delta CVD 4h
            df_4h_closed['single_delta'] = np.where(df_4h_closed['c'] > df_4h_closed['o'], df_4h_closed['v'], -df_4h_closed['v'])
            smoothed_delta_4h = df_4h_closed['single_delta'].ewm(span=5, adjust=False).mean().iloc[-1]
            
            if smoothed_delta_4h > (recent_vol_4h * 0.1):
                delta_4h[s] = "BUY"
            elif smoothed_delta_4h < -(recent_vol_4h * 0.1):
                delta_4h[s] = "SELL"
            else:
                delta_4h[s] = "NEUTRAL"
                
        except:
            continue
            
    # 2. Calcul du Signal de Cohésion pour le 15m (Tactique)
    df_close_15m = pd.DataFrame(data_15m).dropna()
    weights_15m = {s: 0.0 for s in symbols}
    sig_15m = 0.0
    
    if len(df_close_15m) > 5:
        returns_15m = np.log(df_close_15m / df_close_15m.shift(1)).dropna()
        vals_15m, vecs_15m = np.linalg.eigh(returns_15m.corr())
        sig_15m = vals_15m[-1]
        for idx, col in enumerate(df_close_15m.columns):
            weights_15m[col] = np.abs(vecs_15m[idx, -1])
            
    # 3. Calcul du Signal de Cohésion pour le 4h (Macro)
    df_close_4h = pd.DataFrame(data_4h).dropna()
    weights_4h = {s: 0.0 for s in symbols}
    sig_4h = 0.0
    
    if len(df_close_4h) > 5:
        returns_4h = np.log(df_close_4h / df_close_4h.shift(1)).dropna()
        vals_4h, vecs_4h = np.linalg.eigh(returns_4h.corr())
        sig_4h = vals_4h[-1]
        for idx, col in enumerate(df_close_4h.columns):
            weights_4h[col] = np.abs(vecs_4h[idx, -1])
            
    # 4. Calcul ATR pour le Focus (toujours basé sur l'heure fermée)
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
        
    # 5. Données Macro
    try:
        global_assets = {'BTC-USD': 'BTC', '^IXIC': 'Nasdaq'}
        m_data = yf.download(list(global_assets.keys()), period="1mo", progress=False)['Close'].dropna()
        m_corr = np.log(m_data / m_data.shift(1)).dropna().corr()
        corr_val = m_corr.loc['BTC-USD', '^IXIC']
    except:
        corr_val = 0.0
        
    return sig_15m, weights_15m, vols_15m, delta_15m, sig_4h, weights_4h, vols_4h, delta_4h, price_focus, atr_pct, corr_val

# Exécution des calculs
with st.spinner("Analyse quantitative double-timeframe en cours..."):
    sig_15m, weights_15m, vols_15m, delta_15m, sig_4h, weights_4h, vols_4h, delta_4h, price_focus, atr_pct, corr_val = fetch_and_calculate_dual(selected_symbols, focus_symbol)

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
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(label="Cohésion Tactique (15m)", value=f"{sig_15m:.2f}")
with col_b:
    st.metric(label="Cohésion Macro (4h)", value=f"{sig_4h:.2f}")
with col_c:
    st.metric(label="Corrélation Nasdaq", value=f"{corr_val:.2f}")

# --- SELECTION DES DONNEES SELON L'AFFICHAGE DU SMARTPHONE ---
if timeframe_view == '15m':
    current_weights = weights_15m
    current_vols = vols_15m
    current_deltas = delta_15m
    view_label = "TACTIQUE (15m)"
else:
    current_weights = weights_4h
    current_vols = vols_4h
    current_deltas = delta_4h
    view_label = "MACRO (4h)"

# --- GRAPHICS COMPATIBLES STREAMLIT ---
st.subheader(f"📈 Graphiques de Force - Vue {view_label}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Graphique de Gauche : Signal MTF (Weights)
weights_y = [current_weights.get(s, 0) for s in selected_symbols]
ax1.barh(selected_symbols, weights_y, color=['#FF5722' if focus_symbol in s else '#3399FF' for s in selected_symbols])
ax1.set_title(f"SIGNAL MTF (Cohésion : {sig_15m if timeframe_view == '15m' else sig_4h:.2f})")

# Graphique de Droite : RVOL & Delta
colors_rvol = []
for s in selected_symbols:
    d = current_deltas.get(s, "NEUTRAL")
    if d == "BUY":
        colors_rvol.append('#4CAF50')
    elif d == "SELL":
        colors_rvol.append('#F44336')
    else:
        colors_rvol.append('#9E9E9E')

rvols_y = [current_vols.get(s, 0) for s in selected_symbols]
ax2.barh(selected_symbols, rvols_y, color=colors_rvol)
ax2.axvline(1.0, color='black', linestyle='--')
ax2.set_title(f"VOLUME RELATIF & AGRESSIVITÉ {view_label}")

st.pyplot(fig)
