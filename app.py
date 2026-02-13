import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import concurrent.futures
import time
import pytz
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nselib
from nselib import capital_market
import numpy as np

# --- NLTK SETUP (Auto-Download) ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar v22.0", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"
INSTITUTIONAL_FILE = "institutional_history.json"

# --- PRO CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    
    /* Metric Cards */
    .metric-container { 
        background-color: #1E1E1E; 
        border: 1px solid #333; 
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
    }
    .metric-value { font-size: 26px; font-weight: bold; margin: 5px 0; color: #FFF; }
    
    /* Verdict Card */
    .verdict-card { background: linear-gradient(45deg, #1e1e1e, #2d2d2d); padding: 20px; border-radius: 12px; border-left: 6px solid; margin-bottom: 20px; text-align:center; }
    
    /* Table Styling */
    .stTable { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except: return default
    return default

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

if 'watchlist' not in st.session_state: st.session_state.watchlist = load_json(WATCHLIST_FILE, {"india": [], "global": []})
if 'trading' not in st.session_state: st.session_state.trading = load_json(TRADING_FILE, {"india": {"cash": 1000000.0, "holdings": {}}, "global": {"cash": 100000.0, "holdings": {}}})

# --- BACKEND FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_nifty50_tickers():
    fallback_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"]
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url)
        for table in tables:
            if 'Symbol' in table.columns: return [f"{s}.NS" for s in table['Symbol'].tolist()]
        return fallback_list
    except: return fallback_list

@st.cache_data(ttl=300)
def get_ticker_data_parallel(tickers):
    def fetch(t):
        try:
            s = yf.Ticker(t)
            h = s.history(period="5d", interval="1d")
            if len(h) > 1:
                return {
                    "symbol": t, 
                    "price": h['Close'].iloc[-1], 
                    "change": ((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2])*100
                }
        except: return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [r for r in executor.map(fetch, tickers) if r]
    return results

# --- ALPHA ENGINES (SMART MONEY & MACRO) ---

@st.cache_data(ttl=3600)
def get_fii_dii_activity():
    try:
        data = capital_market.fii_dii_trading_activity()
        if not data.empty:
            fii_buy = float(data[data['Category'] == 'FII/FPI *']['Buy Value'].iloc[0].replace(',',''))
            fii_sell = float(data[data['Category'] == 'FII/FPI *']['Sell Value'].iloc[0].replace(',',''))
            dii_buy = float(data[data['Category'] == 'DII **']['Buy Value'].iloc[0].replace(',',''))
            dii_sell = float(data[data['Category'] == 'DII **']['Sell Value'].iloc[0].replace(',',''))
            return {"fii_net": fii_buy - fii_sell, "dii_net": dii_buy - dii_sell, "date": data['Date'].iloc[0]}
    except: return {"fii_net": 0, "dii_net": 0, "date": "N/A"}

@st.cache_data(ttl=3600)
def get_macro_environment():
    tickers = ["^TNX", "DX-Y.NYB", "CL=F", "^NSEI"]
    data = yf.download(tickers, period="6mo", interval="1d")['Close']
    res = {}
    if not data.empty:
        if "^TNX" in data:
            tnx_curr = data["^TNX"].iloc[-1]
            tnx_prev = data["^TNX"].iloc[-20]
            res['us10y'] = {"val": tnx_curr, "trend": "RISING" if tnx_curr > tnx_prev else "FALLING"}
        if "CL=F" in data: res['oil'] = {"val": data["CL=F"].iloc[-1]}
        if "^NSEI" in data:
            nifty = data["^NSEI"]
            sma_200 = nifty.rolling(window=200).mean().iloc[-1]
            res['nifty_trend'] = "BULLISH" if nifty.iloc[-1] > sma_200 else "BEARISH"
    return res

@st.cache_data(ttl=1800)
def get_sector_rotation_map():
    sectors = {"Auto": "^CNXAUTO", "Bank": "^NSEBANK", "IT": "^CNXIT", "Metal": "^CNXMETAL", "Pharma": "^CNXPHARMA", "FMCG": "^CNXFMCG", "Energy": "^CNXENERGY"}
    try:
        tickers = list(sectors.values()) + ["^NSEI"]
        data = yf.download(tickers, period="6mo", interval="1d")['Close']
        if data.empty or "^NSEI" not in data.columns: return pd.DataFrame()

        rrg_data = []
        nifty = data["^NSEI"]
        for name, ticker in sectors.items():
            if ticker in data.columns:
                rs_raw = data[ticker] / nifty
                rs_trend = ((rs_raw.iloc[-1] - rs_raw.rolling(20).mean().iloc[-1]) / rs_raw.rolling(20).mean().iloc[-1]) * 100
                rs_mom = ((rs_raw.iloc[-1] - rs_raw.iloc[-10]) / rs_raw.iloc[-10]) * 100
                
                status = "LAGGING"
                if rs_trend > 0 and rs_mom > 0: status = "LEADING"
                elif rs_trend > 0 and rs_mom < 0: status = "WEAKENING"
                elif rs_trend < 0 and rs_mom > 0: status = "IMPROVING"
                
                rrg_data.append({"Sector": name, "RS_Trend": rs_trend, "RS_Momentum": rs_mom, "Status": status})
        return pd.DataFrame(rrg_data)
    except: return pd.DataFrame()

# --- FORENSIC & VALUATION ENGINE (NEW v22.0) ---

@st.cache_data(ttl=86400)
def get_forensic_analysis(ticker):
    """Deep dives into Cash Flow quality and Valuation"""
    try:
        s = yf.Ticker(ticker)
        # Fetch Financials
        cf = s.cashflow
        fin = s.financials
        info = s.info
        
        # Safe Defaults
        quality_ratio = 0
        fcf = 0
        roe = info.get('returnOnEquity', 0)
        pe = info.get('trailingPE', 0)
        debt_eq = info.get('debtToEquity', 0)
        peg = info.get('pegRatio', 0)
        
        # Forensic Calculations
        if not cf.empty and not fin.empty:
            # 1. Earnings Quality (CFO / Net Income)
            try:
                # Handle different key names in yfinance
                ocf = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else cf.loc['Total Cash From Operating Activities'].iloc[0]
                ni = fin.loc['Net Income'].iloc[0]
                quality_ratio = ocf / ni if ni != 0 else 0
            except: pass
            
            # 2. Free Cash Flow
            try:
                fcf = cf.loc['Free Cash Flow'].iloc[0]
            except: 
                # Fallback calculation
                try:
                    ocf = cf.loc['Operating Cash Flow'].iloc[0]
                    capex = cf.loc['Capital Expenditure'].iloc[0]
                    fcf = ocf + capex # Capex is usually negative
                except: pass
                
        return {
            "quality_ratio": quality_ratio,
            "fcf": fcf,
            "roe": roe * 100 if roe else 0,
            "pe": pe,
            "debt_eq": debt_eq,
            "peg": peg
        }
    except: return None

# --- RISK COMMANDER (DEFENSE INTELLIGENCE) ---

@st.cache_data(ttl=3600)
def get_refined_risk_intelligence(holdings):
    if not holdings: return None
    tickers = list(holdings.keys())
    # 1 Year data for robust stats
    data = yf.download(tickers, period="1y")['Close']
    if data.empty: return None
    
    returns = data.pct_change().dropna()
    
    # Calculate Weights
    prices = {t: data[t].iloc[-1] for t in tickers}
    values = {t: holdings[t]['qty'] * prices[t] for t in tickers}
    total_val = sum(values.values())
    if total_val == 0: return None
    
    weights = np.array([values[t] / total_val for t in tickers])
    
    # 1. Correlation Matrix & Score
    corr_matrix = returns.corr()
    avg_corr = (corr_matrix.values.sum() - len(tickers)) / (len(tickers)**2 - len(tickers)) if len(tickers) > 1 else 1.0
    
    # 2. Portfolio Volatility (Annualized)
    cov_matrix = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance) * 100
    
    # 3. VaR (99% Confidence, 1 Day)
    var_99 = total_val * (port_volatility / 100 / np.sqrt(252)) * 2.33
    
    return {
        "corr": corr_matrix, "avg_corr": avg_corr,
        "vol": port_volatility, "var": var_99, "total_val": total_val
    }

def generate_market_verdict(macro, fii_data):
    score = 0; reasons = []
    
    # 1. Macro Logic
    us_yield = macro.get('us10y', {}).get('val', 0)
    if us_yield > 0 and us_yield < 4.0: score += 2; reasons.append("✅ US Yields Benign (<4%)")
    elif us_yield > 4.5: score -= 2; reasons.append("❌ US Yields Spiking (>4.5%)")
    
    oil_val = macro.get('oil', {}).get('val', 80)
    if oil_val < 80: score += 1; reasons.append("✅ Crude Oil Stable")
    
    # 2. Smart Money Logic
    if fii_data['fii_net'] > 500: score += 2; reasons.append("✅ FIIs Buying Aggressively")
    elif fii_data['fii_net'] < -500: score -= 2; reasons.append("❌ FIIs Selling Heavily")
    
    # 3. Technical Logic
    if macro.get('nifty_trend') == "BULLISH": score += 2; reasons.append("✅ Nifty above 200 SMA")
    else: score -= 2; reasons.append("⚠️ Nifty below 200 SMA")
    
    if score >= 4: return "STRONG BUY", "#00C805", reasons
    elif score >= 1: return "ACCUMULATE", "#FFA726", reasons
    elif score >= -2: return "NEUTRAL / CAUTION", "#FFD700", reasons
    else: return "STRONG SELL", "#FF3B30", reasons

# --- RENDERERS ---
def render_metric_card(title, value, sub_value, color):
    st.markdown(f"""
    <div class="metric-container" style="border-left: 4px solid {color};">
        <div style="font-size:12px; color:#aaa; font-weight:bold;">{title}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div style="font-size:14px; color:#ddd;">{sub_value}</div>
    </div>""", unsafe_allow_html=True)

# --- APP LAYOUT ---
c_title, c_badge = st.columns([4,1])
with c_title:
    st.title("🦅 Executive Market Radar v22.0")
    st.caption("The '1%' System | Smart Money | X-Ray | Forensics")

# --- GLOBAL FETCH ---
macro_data = get_macro_environment()
fii_data = get_fii_dii_activity()
verdict, v_color, v_reasons = generate_market_verdict(macro_data, fii_data)

st.markdown(f"""
<div class="verdict-card" style="border-color: {v_color};">
    <h2 style="margin:0; color: {v_color};">{verdict}</h2>
    <p style="margin:0; color: #ddd;">System Confidence Score</p>
</div>""", unsafe_allow_html=True)

# --- TABS ---
tabs = st.tabs(["🏛️ Smart Money", "🔭 Macroscope", "🩻 X-Ray", "🛡️ Risk Commander", "🇮🇳 India", "🌎 Global", "📉 Trading", "🧠 Lab (Forensics)"])

# --- TAB 1: SMART MONEY ---
with tabs[0]:
    st.subheader("🏦 Institutional Cash Flow")
    c1, c2, c3, c4 = st.columns(4)
    fii_col = "#00C805" if fii_data['fii_net'] > 0 else "#FF3B30"
    dii_col = "#00C805" if fii_data['dii_net'] > 0 else "#FF3B30"
    with c1: render_metric_card("FII NET FLOW", f"₹{fii_data['fii_net']:.0f} Cr", f"Date: {fii_data['date']}", fii_col)
    with c2: render_metric_card("DII NET FLOW", f"₹{fii_data['dii_net']:.0f} Cr", "Domestic Support", dii_col)
    with c3: render_metric_card("US 10Y YIELD", f"{macro_data.get('us10y', {}).get('val', 0):.2f}%", "Liquidity Proxy", "#FFA726")
    with c4: st.write("### 🧠 Verdict Logic"); st.write(v_reasons)

# --- TAB 2: MACROSCOPE ---
with tabs[1]:
    st.subheader("🌍 Macro Environment")
    if st.button("Load Macro Correlations"):
        try:
            df_macro = yf.download(["^NSEI", "^TNX"], period="2y", interval="1wk")['Close']
            df_norm = df_macro / df_macro.iloc[0] * 100
            st.line_chart(df_norm)
            st.info("Inverse Correlation: When TNX (Red) spikes, Nifty (Blue) often struggles.")
        except: st.error("Data fetch failed.")

# --- TAB 3: X-RAY ---
with tabs[2]:
    st.subheader("🩻 Sector Rotation (RRG Proxy)")
    rrg_df = get_sector_rotation_map()
    if not rrg_df.empty:
        fig_rrg = px.scatter(rrg_df, x="RS_Momentum", y="RS_Trend", color="Status", text="Sector", size=[15]*len(rrg_df),
            color_discrete_map={"LEADING": "#00C805", "WEAKENING": "#FFFF00", "LAGGING": "#FF3B30", "IMPROVING": "#0000FF"},
            title="Sector Relative Strength & Momentum")
        fig_rrg.add_hline(y=0, line_dash="dash", line_color="gray"); fig_rrg.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_rrg, use_container_width=True)

# --- TAB 4: RISK COMMANDER ---
with tabs[3]:
    st.subheader("🛡️ Defense Intelligence (VaR & Hedging)")
    risk_intel = get_refined_risk_intelligence(st.session_state.trading['india']['holdings'])
    
    if risk_intel:
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card("Portfolio Volatility", f"{risk_intel['vol']:.2f}%", "Market Avg: ~15%", "#FFA726")
        with c2: render_metric_card("VaR (99% Confidence)", f"₹{risk_intel['var']:,.0f}", "Max Daily Loss", "#FF3B30")
        with c3: render_metric_card("Diversification Score", f"{(1 - risk_intel['avg_corr']) * 100:.1f}/100", "Higher is Better", "#00C805")
        
        st.divider()
        st.subheader("🛡️ The Shield Strategy")
        if risk_intel['vol'] > 20 or verdict == "STRONG SELL":
            st.warning("🚨 RISK ALERT: High Exposure. Recommended Hedge:")
            h1, h2 = st.columns(2)
            with h1: st.info(f"🏆 **Gold:** Buy ~₹{risk_intel['total_val']*0.15:,.0f} (15%)")
            with h2: st.info(f"📉 **Puts:** Buy {max(1, round(risk_intel['total_val']/25000))} Nifty Lots")
        else: st.success("✅ **Status:** Portfolio structure is stable.")
        
        st.markdown("### 🧬 Correlation Matrix"); st.caption("Values > 0.80 mean stocks move identically (Bad).")
        st.plotly_chart(px.imshow(risk_intel['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1, 1]), use_container_width=True)
    else: st.info("⚠️ Add stocks to 'Trading Floor' to enable Risk Intelligence.")

# --- TAB 5: INDIA PULSE ---
with tabs[4]:
    st.subheader("🇮🇳 Market Pulse")
    tickers = ["^NSEI", "^BSESN", "^NSEBANK", "GC=F"] 
    data = get_ticker_data_parallel(tickers)
    cols = st.columns(4)
    if data:
        for i, d in enumerate(data):
            c_val = "#00C805" if d['change'] >= 0 else "#FF3B30"
            with cols[i]: render_metric_card(d['symbol'], f"{d['price']:,.0f}", f"{d['change']:+.2f}%", c_val)

# --- TAB 6: GLOBAL PULSE ---
with tabs[5]:
    st.subheader("🌍 Global Pulse")
    tickers = ["^GSPC", "^IXIC", "BTC-USD", "EURUSD=X"]
    data = get_ticker_data_parallel(tickers)
    cols = st.columns(4)
    if data:
        for i, d in enumerate(data):
            c_val = "#00C805" if d['change'] >= 0 else "#FF3B30"
            with cols[i]: render_metric_card(d['symbol'], f"{d['price']:,.2f}", f"{d['change']:+.2f}%", c_val)

# --- TAB 7: TRADING FLOOR ---
with tabs[6]:
    st.subheader("📈 Virtual Exchange")
    port = st.session_state.trading['india']
    c1, c2 = st.columns([1,2])
    with c1:
        st.metric("Cash", f"₹{port['cash']:,.0f}")
        ticker = st.text_input("Ticker", "RELIANCE.NS").upper()
        if st.button("BUY"):
            d = yf.Ticker(ticker).history(period="1d")
            if not d.empty:
                price = d['Close'].iloc[-1]; cost = price * 10
                if port['cash'] >= cost:
                    port['cash'] -= cost
                    if ticker in port['holdings']: port['holdings'][ticker]['qty'] += 10
                    else: port['holdings'][ticker] = {'qty': 10, 'avg': price}
                    save_json(TRADING_FILE, st.session_state.trading); st.success(f"Bought 10 {ticker}"); st.rerun()
                else: st.error("No Funds")
    with c2:
        if port['holdings']: st.dataframe(pd.DataFrame(port['holdings']).T)

# --- TAB 8: ANALYST LAB (FORENSICS & VALUATION) ---
with tabs[7]:
    st.subheader("🧠 Analyst Lab: Forensic Research")
    t_input = st.text_input("Deep Analyze Ticker", "RELIANCE.NS", key="forensic_tick")
    
    if st.button("Run Forensic Scan"):
        st.caption("Crunching Cash Flows & Balance Sheets...")
        f_data = get_forensic_analysis(t_input)
        
        if f_data:
            c1, c2, c3 = st.columns(3)
            # 1. Earnings Quality (Forensic Check)
            q_col = "#00C805" if f_data['quality_ratio'] > 1.0 else "#FF3B30"
            with c1:
                render_metric_card("Earnings Quality", f"{f_data['quality_ratio']:.2f}x", "Goal: >1.0 (Cash > Profit)", q_col)
            
            # 2. Free Cash Flow (Valuation Driver)
            fcf_cr = f_data['fcf'] / 10000000 if f_data['fcf'] else 0
            with c2:
                render_metric_card("Free Cash Flow", f"₹{fcf_cr:,.0f} Cr", "Actual Cash Available", "#00C805")
                
            # 3. PEG Ratio (Growth Valuation)
            peg_col = "#00C805" if 0 < f_data['peg'] < 1.5 else "#FF3B30"
            with c3:
                render_metric_card("PEG Ratio", f"{f_data['peg']:.2f}", "Undervalued if < 1.0", peg_col)
            
            st.divider()
            
            # COMPETITIVE WAR ROOM
            st.subheader("⚔️ Competitive War Room (Peer Comparison)")
            st.caption("Relative Valuation Matrix")
            
            peer_stats = {
                "Metric": ["P/E Ratio", "Return on Equity (ROE %)", "Debt to Equity"],
                f"{t_input}": [f"{f_data['pe']:.2f}", f"{f_data['roe']:.2f}%", f"{f_data['debt_eq']:.2f}"],
                "Sector Benchmark": ["25.0", "15.0%", "50.0"] # Simplified benchmark
            }
            st.table(pd.DataFrame(peer_stats))
            
            # MANAGEMENT AUDIT
            st.divider()
            st.subheader("🏛️ Management Audit")
            if f_data['quality_ratio'] < 0.8:
                st.error("⚠️ **RED FLAG:** Company is reporting profits but generating weak cash flow. Possible aggressive accounting.")
            elif f_data['debt_eq'] > 100:
                st.warning("⚠️ **Leverage Alert:** Debt is higher than Equity. Check interest coverage.")
            else:
                st.success("✅ **Clean Chit:** Financials look robust. Strong cash conversion and manageable debt.")
        else: st.error("Could not fetch deep financial data. Ticker might be incorrect or data unavailable.")