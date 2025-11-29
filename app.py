import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime
import math
import json
import os
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar 14.0 (Stable)", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"
TRANSACTION_FILE = "transactions.json"

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-container { background-color: #1E1E1E; border: 1px solid #333; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
    .metric-value { font-size: 24px; font-weight: bold; margin: 2px 0; }
    .verdict-pass { color: #4CAF50; border: 1px solid #4CAF50; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .verdict-fail { color: #FF5252; border: 1px solid #FF5252; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .news-card { border-left: 3px solid #4CAF50; background-color: #262730; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .news-title { font-size: 14px; font-weight: 600; color: #E0E0E0; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except: return default # Return default if file is corrupted
    return default

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

if 'watchlist' not in st.session_state: st.session_state.watchlist = load_json(WATCHLIST_FILE, {"india": [], "global": []})
if 'trading' not in st.session_state: st.session_state.trading = load_json(TRADING_FILE, {"india": {"cash": 1000000.0, "holdings": {}}, "global": {"cash": 100000.0, "holdings": {}}})
if 'transactions' not in st.session_state: st.session_state.transactions = load_json(TRANSACTION_FILE, [])

# --- SAFE BACKEND FUNCTIONS (ANTI-CRASH) ---

def get_google_rss(query): 
    return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

def safe_float(val):
    """Safely converts to float, returns 0.0 if failed"""
    try:
        if val is None: return 0.0
        return float(val)
    except: return 0.0

@st.cache_data(ttl=300)
def get_ticker_data_serial(tickers):
    """Fetches tickers one-by-one to avoid Rate Limiting errors"""
    data = []
    if not tickers: return []
    
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            # Fetch minimal data to be fast
            hist = stock.history(period="5d")
            
            if len(hist) > 1:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                data.append({
                    "symbol": t, 
                    "price": curr, 
                    "change": chg, 
                    "high": hist['High'].iloc[-1], 
                    "low": hist['Low'].iloc[-1], 
                    "hist": hist['Close'].tolist()
                })
        except Exception: 
            continue # Skip bad tickers silently
    return data

@st.cache_data(ttl=600)
def fetch_news(url_list):
    news_items = []
    for url in url_list:
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'bozo') and feed.bozo == 1: continue
            
            for entry in feed.entries[:3]:
                blob = TextBlob(entry.title)
                pol = blob.sentiment.polarity
                color = "🟢" if pol > 0.1 else "🔴" if pol < -0.1 else "⚪"
                news_items.append({
                    "title": entry.title, 
                    "link": entry.link, 
                    "source": entry.source.title if 'source' in entry else "News", 
                    "date": entry.published[:17] if 'published' in entry else "Recent", 
                    "mood": color
                })
        except: continue
    return news_items

@st.cache_data(ttl=3600)
def get_company_deep_dive(ticker):
    """Heavy function cached for 1 hour"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.history(period="1y")
    except: return None, None

# --- RENDERERS ---
def render_metrics(data):
    if not data: st.caption("Data unavailable or loading..."); return
    cols = st.columns(len(data)) if len(data) <= 4 else st.columns(4)
    for i, d in enumerate(data):
        col = cols[i % 4] if i >= 4 else cols[i]
        with col:
            c = "#00C805" if d['change'] >= 0 else "#FF3B30"
            bg = "rgba(0,200,5,0.1)" if d['change'] >=0 else "rgba(255,59,48,0.1)"
            
            # Range Bar Logic
            denom = d['high'] - d['low']
            rng = 50
            if denom > 0: rng = ((d['price'] - d['low']) / denom) * 100
            
            st.markdown(f"""
            <div class="metric-container" style="border-left: 4px solid {c}; background: linear-gradient(180deg, #1E1E1E 0%, {bg} 100%);">
                <div style="font-size:13px;color:#aaa;">{d['symbol']}</div>
                <div class="metric-value" style="color: {c}">{d['price']:,.2f}</div>
                <div style="font-size:13px;font-weight:600;color: {c}">{d['change']:+.2f}%</div>
                <div style="margin-top: 8px; height:4px; background:#333; position:relative;">
                    <div style="width:{rng}%; height:100%; background:{c}; position:absolute;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

def render_news_ui(news):
    if not news: st.caption("No recent news."); return
    for n in news:
        st.markdown(f"""<div class="news-card"><a href="{n['link']}" class="news-title" target="_blank">{n['title']}</a><br><small style="color:#888">{n['mood']} {n['source']} • {n['date']}</small></div>""", unsafe_allow_html=True)

# --- TRADING LOGIC ---
def execute_trade(market, action, ticker, qty, price):
    port = st.session_state.trading[market]
    val = price * qty
    fee = val * 0.001
    cost = val + fee
    
    if action == "BUY":
        if cost > port["cash"]: st.error("❌ Insufficient Funds"); return
        port["cash"] -= cost
        if ticker in port["holdings"]:
            old = port["holdings"][ticker]
            new_qty = old["qty"] + qty
            new_avg = ((old["qty"] * old["avg_price"]) + cost) / new_qty
            port["holdings"][ticker] = {"qty": new_qty, "avg_price": new_avg}
        else:
            port["holdings"][ticker] = {"qty": qty, "avg_price": cost/qty}
        st.success(f"BOUGHT {qty} {ticker}")
        
    elif action == "SELL":
        if ticker not in port["holdings"] or port["holdings"][ticker]["qty"] < qty: st.error("❌ Invalid Sell"); return
        port["cash"] += (val - fee)
        port["holdings"][ticker]["qty"] -= qty
        if port["holdings"][ticker]["qty"] <= 0: del port["holdings"][ticker]
        st.success(f"SOLD {qty} {ticker}")

    # Log
    st.session_state.transactions.insert(0, {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "type": action, 
        "symbol": ticker, "qty": qty, "price": price, "fee": fee
    })
    save_json(TRANSACTION_FILE, st.session_state.transactions)
    save_json(TRADING_FILE, st.session_state.trading)
    time.sleep(1) # Wait for file write
    st.rerun()

# --- APP LAYOUT ---
st.title("🦅 Executive Market Radar 14.0 (Stable)")
st.caption("Crash-Proof Engine | Serial Data Fetching | Deep Analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "📈 Trade", "🧠 Analyst"])

# --- TAB 1: INDIA ---
with tab1:
    st.markdown("### 📊 Market Pulse")
    render_metrics(get_ticker_data_serial(["^NSEI", "^BSESN", "^NSEBANK"]))
    
    if st.session_state.watchlist["india"]:
        st.markdown("### ⭐ Watchlist")
        render_metrics(get_ticker_data_serial(st.session_state.watchlist["india"]))
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.caption("Startups & Tech"); render_news_ui(fetch_news([get_google_rss("Indian Startup Funding")]))
    with c2: st.caption("RBI & Banking"); render_news_ui(fetch_news([get_google_rss("RBI India Policy")]))

# --- TAB 2: GLOBAL ---
with tab2:
    st.markdown("### 🌍 Global Pulse")
    render_metrics(get_ticker_data_serial(["^GSPC", "^IXIC", "GC=F"]))
    if st.session_state.watchlist["global"]:
        st.markdown("### ⭐ Watchlist")
        render_metrics(get_ticker_data_serial(st.session_state.watchlist["global"]))
    st.divider()
    render_news_ui(fetch_news(["https://www.cnbc.com/id/100003114/device/rss/rss.html"]))

# --- TAB 3: CEO RADAR ---
with tab3:
    st.markdown("### 🏛️ Strategic Risk")
    c1, c2 = st.columns(2)
    
    # Safe Fetching for Macros
    macros = get_ticker_data_serial(["^TNX", "HG=F", "GC=F"])
    macro_map = {d['symbol']: d['price'] for d in macros}
    
    with c1:
        tnx = macro_map.get("^TNX", 0)
        st.metric("10Y US Yield", f"{tnx:.2f}%", delta="Risk-Free Rate")
    with c2:
        cu = macro_map.get("HG=F", 0)
        au = macro_map.get("GC=F", 1) # Avoid div by zero
        st.metric("Copper/Gold Ratio", f"{(cu/au)*1000:.2f}", delta="Growth Metric")
    
    st.divider()
    st.markdown("### 🔥 Sector Heatmap (India)")
    sec_proxies = {"Bank": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Energy": "^CNXENERGY"}
    sec_data = get_ticker_data_serial(list(sec_proxies.values()))
    
    if sec_data:
        smap = {d['symbol']: d['change'] for d in sec_data}
        df_sec = pd.DataFrame([{"Sector": k, "Change": smap.get(v, 0)} for k,v in sec_proxies.items()])
        fig = px.bar(df_sec, x='Sector', y='Change', color='Change', color_continuous_scale=['#FF5252', '#333333', '#4CAF50'], range_color=[-2, 2])
        fig.update_layout(height=250, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: TRADING ---
with tab4:
    st.markdown("### 📈 Virtual Floor")
    mkt = st.radio("Market", ["🇮🇳 India", "🇺🇸 Global"], horizontal=True)
    m_key = "india" if "India" in mkt else "global"
    curr = "₹" if "India" in mkt else "$"
    
    port = st.session_state.trading[m_key]
    
    # Valuation
    total_val = 0
    holdings = port["holdings"]
    if holdings:
        live_prices = get_ticker_data_serial(list(holdings.keys()))
        pmap = {d['symbol']: d['price'] for d in live_prices}
        for t, d in holdings.items():
            total_val += d['qty'] * pmap.get(t, d['avg_price'])
            
    c1, c2 = st.columns(2)
    c1.metric("Cash Available", f"{curr}{port['cash']:,.0f}")
    c2.metric("Portfolio Value", f"{curr}{total_val:,.0f}")
    
    # Trade Box
    with st.expander("⚡ Execute Trade", expanded=True):
        t_sym = st.text_input("Ticker", "RELIANCE.NS" if m_key=="india" else "AAPL").upper()
        if t_sym:
            live = get_ticker_data_serial([t_sym])
            if live:
                lp = live[0]['price']
                st.write(f"Live Price: **{curr}{lp:,.2f}**")
                c_act, c_qty, c_btn = st.columns([1,1,1])
                act = c_act.selectbox("Action", ["BUY", "SELL"])
                qty = c_qty.number_input("Qty", 1, 10000)
                if c_btn.button("CONFIRM"): execute_trade(m_key, act, t_sym, qty, lp)
            else: st.error("Ticker not found")

    st.markdown("### 📜 Holdings")
    if holdings:
        data = []
        for t, d in holdings.items():
            data.append({"Ticker": t, "Qty": d['qty'], "Avg Cost": f"{d['avg_price']:.2f}"})
        st.dataframe(pd.DataFrame(data), use_container_width=True)

# --- TAB 5: ANALYST LAB (CRASH PROOF) ---
with tab5:
    st.markdown("### 🔍 Deep Dive")
    c_in, c_meth = st.columns([1, 2])
    ticker = c_in.text_input("Ticker", "TCS.NS").upper()
    method = c_meth.selectbox("Method", ["🚀 Growth (CAN SLIM)", "🪄 Value (Magic Formula)", "🏰 MOAT Analysis"])
    
    if ticker:
        info, hist = get_company_deep_dive(ticker)
        if info and not hist.empty:
            curr = hist['Close'].iloc[-1]
            st.metric(f"{info.get('shortName', ticker)}", f"{curr:,.2f}")
            
            # --- SAFE SCORECARD ENGINE ---
            score = 0
            # Helper to get safely
            def get_val(key, default=0): return safe_float(info.get(key, default))
            
            if "CAN SLIM" in method:
                eps_g = get_val('earningsGrowth')
                rev_g = get_val('revenueGrowth')
                high52 = get_val('fiftyTwoWeekHigh')
                
                c1, c2, c3 = st.columns(3)
                c1.metric("EPS Growth", f"{eps_g*100:.1f}%", delta="Target > 15%")
                c2.metric("Rev Growth", f"{rev_g*100:.1f}%", delta="Target > 15%")
                
                dist = 0
                if high52 > 0: dist = (curr / high52) * 100
                c3.metric("Vs 52W High", f"{dist:.0f}%", delta="Target > 85%")
                
                if eps_g > 0.15: score += 1
                if rev_g > 0.15: score += 1
                if dist > 85: score += 1
                
            elif "Magic Formula" in method:
                pe = get_val('trailingPE')
                roe = get_val('returnOnEquity')
                ey = (1/pe * 100) if pe > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.metric("Earnings Yield", f"{ey:.2f}%", delta="Target > 5%")
                c2.metric("ROE", f"{roe*100:.1f}%", delta="Target > 15%")
                
                if ey > 5: score += 1
                if roe > 0.15: score += 1

            elif "MOAT" in method:
                pm = get_val('grossMargins')
                roe = get_val('returnOnEquity')
                de = get_val('debtToEquity')
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Gross Margin", f"{pm*100:.1f}%")
                c2.metric("ROE", f"{roe*100:.1f}%")
                c3.metric("Debt/Eq", f"{de:.0f}%")
                
                if pm > 0.40: score += 1
                if roe > 0.15: score += 1
                if de < 50: score += 1

            # VERDICT
            st.divider()
            if score >= 2: st.markdown(f"<div class='verdict-pass'>✅ PASS ({score}/3)</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div class='verdict-fail'>❌ FAIL ({score}/3)</div>", unsafe_allow_html=True)
            
            # CHART (Subplots)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='teal'), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Data Unavailable or Ticker Invalid")

# --- SIDEBAR WATCHLIST ---
with st.sidebar:
    st.header("📝 Watchlist")
    with st.expander("🇮🇳 India"):
        it = st.text_input("Add IN", key="it").upper()
        if st.button("Add"): 
            st.session_state.watchlist["india"].append(it); save_json(WATCHLIST_FILE, st.session_state.watchlist); st.rerun()
    
    if st.button("🗑️ Reset All Data"):
        if os.path.exists(TRADING_FILE): os.remove(TRADING_FILE)
        if os.path.exists(WATCHLIST_FILE): os.remove(WATCHLIST_FILE)
        st.cache_data.clear()
        st.rerun()