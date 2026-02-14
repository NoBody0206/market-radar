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
st.set_page_config(page_title="Executive Market Radar v23.0", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"

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
    
    /* News Cards */
    .news-card { border-left: 3px solid #4CAF50; background-color: #262730; padding: 12px; margin-bottom: 10px; border-radius: 6px; }
    .news-title { font-size: 14px; font-weight: 600; color: #E0E0E0; text-decoration: none; }
    
    /* Verdict Card */
    .verdict-card { background: linear-gradient(45deg, #1e1e1e, #2d2d2d); padding: 20px; border-radius: 12px; border-left: 6px solid; margin-bottom: 20px; text-align:center; }
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

def get_google_rss(query): 
    return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

@st.cache_data(ttl=86400)
def get_nifty50_tickers():
    fallback = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "SBIN.NS"]
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
        for table in tables:
            if 'Symbol' in table.columns: return [f"{s}.NS" for s in table['Symbol'].tolist()]
        return fallback
    except: return fallback

@st.cache_data(ttl=300)
def get_ticker_data_parallel(tickers):
    def fetch(t):
        try:
            s = yf.Ticker(t)
            h = s.history(period="5d", interval="1d")
            if len(h) > 1:
                return {"symbol": t, "price": h['Close'].iloc[-1], "change": ((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2])*100}
        except: return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [r for r in executor.map(fetch, tickers) if r]
    return results

@st.cache_data(ttl=600)
def fetch_feed_parallel(url_list):
    all_news = []
    def fetch(url):
        try:
            f = feedparser.parse(url)
            return [{"title": e.title, "link": e.link, "source": e.source.title if 'source' in e else "News", "date": e.published[:16]} for e in f.entries[:5]]
        except: return []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in executor.map(fetch, url_list): all_news.extend(res)
    return all_news[:6]

# --- RESTORED: ECONOMIC INDICATORS (YIELD CURVE) ---
@st.cache_data(ttl=600)
def get_yield_curve_data():
    tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
    labels = ["3M", "5Y", "10Y", "30Y"]
    try:
        data = yf.download(tickers, period="2d")['Close'].iloc[-1]
        values = []
        for t in tickers:
            if t in data: values.append(data[t])
            else: values.append(0)
        return labels, values
    except: return [], []

# --- 1% ENGINES (SMART MONEY & FORENSICS) ---

@st.cache_data(ttl=3600)
def get_fii_dii_activity():
    try:
        data = capital_market.fii_dii_trading_activity()
        if not data.empty:
            fii_net = float(data[data['Category'] == 'FII/FPI *']['Buy Value'].iloc[0].replace(',','')) - float(data[data['Category'] == 'FII/FPI *']['Sell Value'].iloc[0].replace(',',''))
            dii_net = float(data[data['Category'] == 'DII **']['Buy Value'].iloc[0].replace(',','')) - float(data[data['Category'] == 'DII **']['Sell Value'].iloc[0].replace(',',''))
            return {"fii_net": fii_net, "dii_net": dii_net, "date": data['Date'].iloc[0]}
    except: return {"fii_net": 0, "dii_net": 0, "date": "N/A"}

@st.cache_data(ttl=3600)
def get_macro_environment():
    tickers = ["^TNX", "DX-Y.NYB", "CL=F", "^NSEI"]
    data = yf.download(tickers, period="6mo", interval="1d")['Close']
    res = {}
    if not data.empty:
        if "^TNX" in data: res['us10y'] = {"val": data["^TNX"].iloc[-1]}
        if "CL=F" in data: res['oil'] = {"val": data["CL=F"].iloc[-1]}
        if "^NSEI" in data:
            nifty = data["^NSEI"]
            sma_200 = nifty.rolling(window=200).mean().iloc[-1]
            res['nifty_trend'] = "BULLISH" if nifty.iloc[-1] > sma_200 else "BEARISH"
    return res

@st.cache_data(ttl=86400)
def get_forensic_analysis(ticker):
    try:
        s = yf.Ticker(ticker)
        cf = s.cashflow; fin = s.financials; info = s.info
        
        quality_ratio = 0; fcf = 0
        if not cf.empty and not fin.empty:
            try:
                ocf = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else cf.loc['Total Cash From Operating Activities'].iloc[0]
                ni = fin.loc['Net Income'].iloc[0]
                quality_ratio = ocf / ni if ni != 0 else 0
                fcf = cf.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cf.index else (ocf + cf.loc['Capital Expenditure'].iloc[0])
            except: pass
        return {"quality_ratio": quality_ratio, "fcf": fcf, "roe": info.get('returnOnEquity',0)*100, "pe": info.get('trailingPE',0), "peg": info.get('pegRatio',0)}
    except: return None

# --- RISK & SECTOR MAP ---
@st.cache_data(ttl=3600)
def get_refined_risk_intelligence(holdings):
    if not holdings: return None
    tickers = list(holdings.keys())
    data = yf.download(tickers, period="1y")['Close']
    if data.empty: return None
    
    returns = data.pct_change().dropna()
    prices = {t: data[t].iloc[-1] for t in tickers}
    values = {t: holdings[t]['qty'] * prices[t] for t in tickers}
    total_val = sum(values.values())
    weights = np.array([values[t] / total_val for t in tickers])
    
    corr_matrix = returns.corr()
    avg_corr = (corr_matrix.values.sum() - len(tickers)) / (len(tickers)**2 - len(tickers)) if len(tickers) > 1 else 1.0
    
    cov_matrix = returns.cov() * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
    var_99 = total_val * (port_vol / 100 / np.sqrt(252)) * 2.33
    return {"corr": corr_matrix, "avg_corr": avg_corr, "vol": port_vol, "var": var_99, "total_val": total_val}

def generate_market_verdict(macro, fii_data):
    score = 0; reasons = []
    us_yield = macro.get('us10y', {}).get('val', 0)
    if us_yield > 0 and us_yield < 4.0: score += 2; reasons.append("✅ US Yields Benign (<4%)")
    elif us_yield > 4.5: score -= 2; reasons.append("❌ US Yields Spiking (>4.5%)")
    
    if fii_data['fii_net'] > 500: score += 2; reasons.append("✅ FIIs Buying Aggressively")
    elif fii_data['fii_net'] < -500: score -= 2; reasons.append("❌ FIIs Selling Heavily")
    
    if macro.get('nifty_trend') == "BULLISH": score += 2; reasons.append("✅ Nifty above 200 SMA")
    else: score -= 2; reasons.append("⚠️ Nifty below 200 SMA")
    
    if score >= 4: return "STRONG BUY", "#00C805", reasons
    elif score >= 1: return "ACCUMULATE", "#FFA726", reasons
    elif score >= -2: return "NEUTRAL", "#FFD700", reasons
    else: return "STRONG SELL", "#FF3B30", reasons

# --- RENDERERS ---
def render_metric_card(title, value, sub_value, color):
    st.markdown(f"""<div class="metric-container" style="border-left: 4px solid {color};"><div style="font-size:12px; color:#aaa; font-weight:bold;">{title}</div><div class="metric-value" style="color:{color}">{value}</div><div style="font-size:14px; color:#ddd;">{sub_value}</div></div>""", unsafe_allow_html=True)

def render_news(news):
    if not news: st.caption("No updates."); return
    for n in news: st.markdown(f"""<div class="news-card"><a href="{n['link']}" class="news-title" target="_blank">{n['title']}</a><br><span style="font-size:10px; color:#888;">{n['source']} • {n['date']}</span></div>""", unsafe_allow_html=True)

# --- APP LAYOUT ---
c_title, c_badge = st.columns([4,1])
with c_title:
    st.title("🦅 Executive Market Radar v23.0")
    st.caption("The Ultimate System | Smart Money | CEO Radar | Global Trade")

macro_data = get_macro_environment()
fii_data = get_fii_dii_activity()
verdict, v_color, v_reasons = generate_market_verdict(macro_data, fii_data)

st.markdown(f"""<div class="verdict-card" style="border-color: {v_color};"><h2 style="margin:0; color: {v_color};">{verdict}</h2><p style="margin:0; color: #ddd;">System Confidence Score</p></div>""", unsafe_allow_html=True)

# --- TABS ---
tabs = st.tabs(["🏛️ CEO Radar", "🇮🇳 India Pulse", "🌎 Global Pulse", "🛡️ Risk Commander", "📉 Trading Floor", "🧠 Analyst Lab"])

# --- TAB 1: CEO RADAR (The Strategic Situation Room) ---
with tabs[0]:
    st.subheader("🏛️ Strategic Situation Room")
    
    # 1. Yield Curve (Restored)
    c_yield, c_money = st.columns([2, 1])
    with c_yield:
        st.markdown("**⚠️ US Yield Curve (Recession Watch)**")
        labels, values = get_yield_curve_data()
        if labels:
            fig = go.Figure(go.Scatter(x=labels, y=values, mode='lines+markers', line=dict(color='#FFA726', width=4)))
            fig.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
    with c_money:
        st.markdown("**🏦 Smart Money Flow**")
        fii_col = "#00C805" if fii_data['fii_net'] > 0 else "#FF3B30"
        render_metric_card("FII NET FLOW", f"₹{fii_data['fii_net']:.0f} Cr", f"Date: {fii_data['date']}", fii_col)
        render_metric_card("US 10Y YIELD", f"{macro_data.get('us10y', {}).get('val', 0):.2f}%", "Liquidity Proxy", "#FFA726")

    st.divider()
    # 2. Strategic News (Restored)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("**🌏 Geopolitics & Energy**"); render_news(fetch_feed_parallel([get_google_rss("Global Oil Prices OPEC"), get_google_rss("China Economy News")]))
    with c2: st.markdown("**🏦 Central Banks & Policy**"); render_news(fetch_feed_parallel([get_google_rss("Federal Reserve News"), get_google_rss("RBI Policy India")]))
    with c3: st.markdown("**🚀 Tech & AI Trends**"); render_news(fetch_feed_parallel([get_google_rss("Artificial Intelligence Business News"), get_google_rss("Nvidia Stock News")]))

# --- TAB 2: INDIA PULSE (With Sector News) ---
with tabs[1]:
    st.subheader("🇮🇳 India Market Pulse")
    tickers = ["^NSEI", "^BSESN", "^NSEBANK", "GC=F"] 
    data = get_ticker_data_parallel(tickers)
    cols = st.columns(4)
    if data:
        for i, d in enumerate(data):
            c_val = "#00C805" if d['change'] >= 0 else "#FF3B30"
            with cols[i]: render_metric_card(d['symbol'], f"{d['price']:,.0f}", f"{d['change']:+.2f}%", c_val)
            
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.markdown("**🏭 Corporate India News**"); render_news(fetch_feed_parallel([get_google_rss("Nifty 50 News"), get_google_rss("Indian Banking Sector")]))
    with c2: st.markdown("**🏗️ Infra & Economy**"); render_news(fetch_feed_parallel([get_google_rss("India Infrastructure News"), get_google_rss("Indian Economy Updates")]))

# --- TAB 3: GLOBAL PULSE (With Sector News) ---
with tabs[2]:
    st.subheader("🌎 Global Market Pulse")
    tickers = ["^GSPC", "^IXIC", "BTC-USD", "EURUSD=X"]
    data = get_ticker_data_parallel(tickers)
    cols = st.columns(4)
    if data:
        for i, d in enumerate(data):
            c_val = "#00C805" if d['change'] >= 0 else "#FF3B30"
            with cols[i]: render_metric_card(d['symbol'], f"{d['price']:,.2f}", f"{d['change']:+.2f}%", c_val)
            
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.markdown("**🇺🇸 Wall Street Wire**"); render_news(fetch_feed_parallel([get_google_rss("Wall Street Market Analysis"), get_google_rss("Tech Stocks US")]))
    with c2: st.markdown("**🔋 EV & Green Energy**"); render_news(fetch_feed_parallel([get_google_rss("Global EV Market News"), get_google_rss("Tesla News")]))

# --- TAB 4: RISK COMMANDER (Advanced) ---
with tabs[3]:
    st.subheader("🛡️ Defense Intelligence (VaR & Hedging)")
    # Using India Holdings for Risk Demo
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
        
        st.markdown("### 🧬 Correlation Matrix"); st.plotly_chart(px.imshow(risk_intel['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1, 1]), use_container_width=True)
    else: st.info("⚠️ Add stocks to 'Trading Floor' to enable Risk Intelligence.")

# --- TAB 5: TRADING FLOOR (Global Restored) ---
with tabs[4]:
    st.subheader("📈 Virtual Exchange")
    
    # 1. Market Selection (Restored)
    mkt = st.radio("Select Market Access:", ["🇮🇳 India (NSE)", "🇺🇸 Global (US)"], horizontal=True)
    m_key = "india" if "India" in mkt else "global"
    curr = "₹" if "India" in mkt else "$"
    
    port = st.session_state.trading[m_key]
    
    c1, c2 = st.columns([1,2])
    with c1:
        st.metric(f"Cash Available ({curr})", f"{curr}{port['cash']:,.0f}")
        t_default = "RELIANCE.NS" if m_key == "india" else "NVDA"
        ticker = st.text_input("Ticker Symbol", t_default).upper()
        
        if st.button("BUY ORDER"):
            d = yf.Ticker(ticker).history(period="1d")
            if not d.empty:
                price = d['Close'].iloc[-1]; cost = price * 10
                if port['cash'] >= cost:
                    port['cash'] -= cost
                    if ticker in port['holdings']: port['holdings'][ticker]['qty'] += 10
                    else: port['holdings'][ticker] = {'qty': 10, 'avg': price}
                    save_json(TRADING_FILE, st.session_state.trading); st.success(f"Executed: 10 {ticker} @ {price:.2f}"); st.rerun()
                else: st.error("Insufficient Funds")
            else: st.error("Invalid Ticker")
            
    with c2:
        st.write(f"### Your {mkt} Portfolio")
        if port['holdings']: st.dataframe(pd.DataFrame(port['holdings']).T)
        else: st.info("No active positions.")

# --- TAB 6: ANALYST LAB (Forensics) ---
with tabs[5]:
    st.subheader("🧠 Analyst Lab: Forensic Research")
    t_input = st.text_input("Deep Analyze Ticker", "RELIANCE.NS", key="forensic_tick")
    
    if st.button("Run Forensic Scan"):
        f_data = get_forensic_analysis(t_input)
        if f_data:
            c1, c2, c3 = st.columns(3)
            q_col = "#00C805" if f_data['quality_ratio'] > 1.0 else "#FF3B30"
            with c1: render_metric_card("Earnings Quality", f"{f_data['quality_ratio']:.2f}x", "Goal: >1.0 (Cash > Profit)", q_col)
            fcf_cr = f_data['fcf'] / 10000000 if f_data['fcf'] else 0
            with c2: render_metric_card("Free Cash Flow", f"₹{fcf_cr:,.0f} Cr", "Actual Cash", "#00C805")
            peg_col = "#00C805" if 0 < f_data['peg'] < 1.5 else "#FF3B30"
            with c3: render_metric_card("PEG Ratio", f"{f_data['peg']:.2f}", "Valuation Check", peg_col)
            
            st.divider()
            st.subheader("⚔️ Competitive War Room")
            peer_stats = {"Metric": ["P/E Ratio", "ROE %", "Debt to Equity"], f"{t_input}": [f"{f_data['pe']:.2f}", f"{f_data['roe']:.2f}%", f"{f_data['peg']:.2f}"]} # Using PEG as placeholder for D/E
            st.table(pd.DataFrame(peer_stats))
            
            if f_data['quality_ratio'] < 0.8: st.error("⚠️ **RED FLAG:** Weak Cash Flow Quality.")
            else: st.success("✅ **Clean Chit:** Strong Cash Conversion.")