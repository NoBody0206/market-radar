import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
import math
import json
import os
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar 12.5", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"
TRANSACTION_FILE = "transactions.json"

# --- PRO CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-container { background-color: #1E1E1E; border: 1px solid #333; border-radius: 10px; padding: 12px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .metric-value { font-size: 24px; font-weight: bold; margin: 2px 0; }
    .method-card { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #64B5F6; margin-bottom: 20px; }
    .score-good { color: #4CAF50; font-weight: bold; }
    .score-bad { color: #FF5252; font-weight: bold; }
    .score-neutral { color: #FFC107; font-weight: bold; }
    .section-header { font-size: 20px; font-weight: 700; margin-top: 20px; margin-bottom: 15px; color: #64B5F6; border-bottom: 1px solid #444; padding-bottom: 5px; }
    .news-card { border-left: 3px solid #4CAF50; background-color: #262730; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .news-title { font-size: 14px; font-weight: 600; color: #E0E0E0; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_json(filename, default):
    if os.path.exists(filename):
        with open(filename, 'r') as f: return json.load(f)
    return default

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

if 'watchlist' not in st.session_state: st.session_state.watchlist = load_json(WATCHLIST_FILE, {"india": [], "global": []})
if 'trading' not in st.session_state: st.session_state.trading = load_json(TRADING_FILE, {"india": {"cash": 1000000.0, "holdings": {}}, "global": {"cash": 100000.0, "holdings": {}}})
if 'transactions' not in st.session_state: st.session_state.transactions = load_json(TRANSACTION_FILE, [])

# --- BACKEND FUNCTIONS ---
def get_google_rss(query): return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

def fetch_single_ticker(t):
    try:
        stock = yf.Ticker(t)
        hist = stock.history(period="5d") 
        if len(hist) > 1:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            chg = ((curr - prev) / prev) * 100
            return {"symbol": t, "price": curr, "change": chg, "high": hist['High'].iloc[-1], "low": hist['Low'].iloc[-1], "hist": hist['Close'].tolist()}
    except: return None

@st.cache_data(ttl=300)
def get_ticker_data_parallel(tickers):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return results

@st.cache_data(ttl=600)
def fetch_feed_parallel(url_list):
    all_news = []
    def fetch_single_feed(url):
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'bozo') and feed.bozo == 1: return []
            items = []
            for entry in feed.entries[:3]:
                blob = TextBlob(entry.title)
                pol = blob.sentiment.polarity
                color = "🟢" if pol > 0.1 else "🔴" if pol < -0.1 else "⚪"
                items.append({"title": entry.title, "link": entry.link, "source": entry.source.title if 'source' in entry else "News", "date": entry.published[:17] if 'published' in entry else "Recent", "mood": color})
            return items
        except: return []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_single_feed, url): url for url in url_list}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: all_news.extend(res)
    return all_news[:10]

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetching all statements safely
        return (stock.info, 
                stock.history(period="1y"), 
                stock.financials, 
                stock.balance_sheet, 
                stock.cashflow)
    except: return None, None, None, None, None

# --- RENDERERS ---
def render_pro_metrics(data_list):
    if not data_list: st.caption("No data."); return
    cols = st.columns(len(data_list)) if len(data_list) <= 4 else st.columns(4)
    for i, d in enumerate(data_list):
        col = cols[i % 4] if i >= 4 else cols[i]
        with col:
            color = "#00C805" if d['change'] >= 0 else "#FF3B30"
            bg = "rgba(0, 200, 5, 0.1)" if d['change'] >= 0 else "rgba(255, 59, 48, 0.1)"
            denom = d['high'] - d['low']
            rng = ((d['price'] - d['low']) / denom) * 100 if denom > 0 else 50
            st.markdown(f"""
            <div class="metric-container" style="border-left: 4px solid {color}; background: linear-gradient(180deg, #1E1E1E 0%, {bg} 100%);">
                <div style="font-size:13px;color:#aaa;">{d['symbol']}</div>
                <div class="metric-value" style="color: {color}">{d['price']:,.2f}</div>
                <div style="font-size:13px;font-weight:600;color: {color}">{d['change']:+.2f}%</div>
                <div style="margin-top: 8px;"><div class="range-container" style="height:4px;background:#333;position:relative"><div style="width:{rng}%;height:100%;background:{color};position:absolute"></div><div style="left:{rng}%;width:2px;height:10px;background:white;position:absolute;top:-3px"></div></div></div>
            </div>""", unsafe_allow_html=True)
            fig = go.Figure(data=go.Scatter(y=d['hist'], mode='lines', fill='tozeroy', line=dict(color=color, width=2), fillcolor=bg))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=35, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

def render_news(news):
    if not news: st.caption("No news."); return
    for n in news: st.markdown(f"""<div class="news-card" style="border-left:3px solid #4CAF50;background:#262730;padding:10px;margin-bottom:8px"><a href="{n['link']}" class="news-title" style="color:#E0E0E0;text-decoration:none;font-weight:600" target="_blank">{n['title']}</a><div style="font-size:11px;color:#888;margin-top:5px"><span>{n['mood']} {n['source']}</span> • <span>{n['date']}</span></div></div>""", unsafe_allow_html=True)

# --- TRADING LOGIC ---
def execute_trade(market, action, ticker, qty, price):
    port = st.session_state.trading[market]
    val = price * qty
    fee = val * 0.001
    cost, rev = val + fee, val - fee
    if action == "BUY":
        if cost > port["cash"]: st.error("❌ Insufficient Funds!"); return
        port["cash"] -= cost
        if ticker in port["holdings"]:
            old_q, old_avg = port["holdings"][ticker]["qty"], port["holdings"][ticker]["avg_price"]
            new_q = old_q + qty
            port["holdings"][ticker] = {"qty": new_q, "avg_price": ((old_q * old_avg) + cost) / new_q}
        else: port["holdings"][ticker] = {"qty": qty, "avg_price": cost/qty}
        st.success(f"✅ BOUGHT {qty} {ticker}")
    elif action == "SELL":
        if ticker not in port["holdings"] or port["holdings"][ticker]["qty"] < qty: st.error("❌ Not enough shares!"); return
        port["cash"] += rev
        port["holdings"][ticker]["qty"] -= qty
        if port["holdings"][ticker]["qty"] == 0: del port["holdings"][ticker]
        st.success(f"✅ SOLD {qty} {ticker}")
    st.session_state.transactions.insert(0, {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "type": action, "symbol": ticker, "qty": qty, "price": price, "fee": fee})
    save_json(TRANSACTION_FILE, st.session_state.transactions); save_json(TRADING_FILE, st.session_state.trading); st.rerun()

# --- MAIN APP ---
st.title("🦅 Executive Market Radar 12.5")
st.caption("Financials in Crores | Stable Engine | 6 Strategy Frameworks")

tab_india, tab_global, tab_ceo, tab_trade, tab_deep = st.tabs(["🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "📈 Trading Floor", "🧠 Analyst Lab"])

# --- TAB 1, 2, 3, 4 (Stable Core) ---
with tab_india:
    st.markdown("<div class='section-header'>📊 Benchmarks</div>", unsafe_allow_html=True)
    render_pro_metrics(get_ticker_data_parallel(["^NSEI", "^BSESN", "^NSEBANK", "USDINR=X"]))
    if st.session_state.watchlist["india"]: render_pro_metrics(get_ticker_data_parallel(st.session_state.watchlist["india"]))
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("**Startups**"); render_news(fetch_feed_parallel([get_google_rss("Indian Startup Funding")]))
    with c2: st.markdown("**Banking**"); render_news(fetch_feed_parallel([get_google_rss("RBI Policy")]))
    with c3: st.markdown("**Corporate**"); render_news(fetch_feed_parallel(["https://www.moneycontrol.com/rss/business.xml"]))

with tab_global:
    st.markdown("<div class='section-header'>🌍 Drivers</div>", unsafe_allow_html=True)
    render_pro_metrics(get_ticker_data_parallel(["^GSPC", "^IXIC", "BTC-USD", "GC=F"]))
    if st.session_state.watchlist["global"]: render_pro_metrics(get_ticker_data_parallel(st.session_state.watchlist["global"]))
    st.divider()
    render_news(fetch_feed_parallel(["https://www.cnbc.com/id/100003114/device/rss/rss.html"]))

with tab_ceo:
    st.markdown("<div class='section-header'>🏛️ Strategic Overview</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚠️ Yield Curve")
        data = get_ticker_data_parallel(["^TNX"])
        if data: st.metric("10Y Yield", f"{data[0]['price']:.2f}%", delta="Borrowing Cost")
    with c2:
        st.subheader("🏗️ Pulse Score")
        rd = get_ticker_data_parallel(["HG=F", "GC=F"])
        pm = {d['symbol']: d['price'] for d in rd}
        if "HG=F" in pm and "GC=F" in pm: st.metric("Copper/Gold", f"{(pm['HG=F']/pm['GC=F'])*1000:.2f}", delta="> 2.0 = Expansion")
    st.divider()
    hm1, hm2 = st.columns(2)
    with hm1:
        st.subheader("🔥 Indian Sectors")
        sec_in = {"Bank": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Energy": "^CNXENERGY"}
        sd_in = get_ticker_data_parallel(list(sec_in.values()))
        if sd_in:
            smap = {d['symbol']: d['change'] for d in sd_in}
            fig = px.bar(pd.DataFrame([{"Sector": k, "Change": smap.get(v, 0)} for k, v in sec_in.items()]), x='Sector', y='Change', color='Change', color_continuous_scale=['#FF5252', '#333333', '#4CAF50'], range_color=[-2, 2])
            fig.update_layout(height=250, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    with hm2:
        st.subheader("🌍 Global Sectors")
        sec_gl = {"Tech": "IXN", "Energy": "IXC", "Finance": "IXG", "Health": "IXJ"}
        sd_gl = get_ticker_data_parallel(list(sec_gl.values()))
        if sd_gl:
            smap_gl = {d['symbol']: d['change'] for d in sd_gl}
            fig2 = px.bar(pd.DataFrame([{"Sector": k, "Change": smap_gl.get(v, 0)} for k, v in sec_gl.items()]), x='Sector', y='Change', color='Change', color_continuous_scale=['#FF5252', '#333333', '#4CAF50'], range_color=[-2, 2])
            fig2.update_layout(height=250, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

with tab_trade:
    st.markdown("<div class='section-header'>📈 Virtual Exchange</div>", unsafe_allow_html=True)
    sub_port, sub_log = st.tabs(["💼 Portfolio", "📜 History"])
    with sub_port:
        mkt = st.radio("Market:", ["🇮🇳 India", "🇺🇸 Global"], horizontal=True)
        m_key = "india" if "India" in mkt else "global"
        curr = "₹" if "India" in mkt else "$"
        sample = "RELIANCE.NS" if "India" in mkt else "TSLA"
        port = st.session_state.trading[m_key]
        holdings = port["holdings"]
        curr_val = 0
        if holdings:
            live = get_ticker_data_parallel(list(holdings.keys()))
            pmap = {d['symbol']: d['price'] for d in live}
            for t, d in holdings.items(): curr_val += d['qty'] * pmap.get(t, d['avg_price'])
        c1, c2 = st.columns(2)
        c1.metric("Cash", f"{curr}{port['cash']:,.0f}")
        c2.metric("Net Worth", f"{curr}{port['cash']+curr_val:,.0f}")
        c_tr, c_hl = st.columns([1, 2])
        with c_tr:
            st.markdown(f"<div class='trade-panel'><b>⚡ Trade ({curr})</b></div>", unsafe_allow_html=True)
            t_sym = st.text_input("Ticker", sample).upper()
            if t_sym:
                ld = get_ticker_data_parallel([t_sym])
                if ld:
                    lp = ld[0]['price']
                    st.metric("Live", f"{curr}{lp:,.2f}")
                    act = st.selectbox("Action", ["BUY", "SELL"])
                    qty = st.number_input("Qty", 1, 1000)
                    if st.button("EXECUTE"): execute_trade(m_key, act, t_sym, qty, lp)
        with c_hl:
            if holdings: st.dataframe(pd.DataFrame([{"Sym": t, "Qty": d['qty'], "Avg": f"{d['avg_price']:.2f}"} for t, d in holdings.items()]), use_container_width=True)
            else: st.info("Empty Portfolio")
    with sub_log: st.dataframe(pd.DataFrame(st.session_state.transactions), use_container_width=True)

# --- TAB 5: ANALYST LAB (WITH FINANCIALS IN CRORES) ---
with tab_deep:
    st.markdown("<div class='section-header'>🔍 Analyst Masterclass</div>", unsafe_allow_html=True)
    
    col_input, col_view = st.columns([2, 1])
    with col_input: ticker = st.text_input("Analyze Ticker:", "RELIANCE.NS").upper()
    with col_view: view_mode = st.radio("View Mode:", ["🧠 Strategy", "📑 Financials"], horizontal=True)

    if ticker:
        info, hist, fin, bal, cash = get_company_info(ticker)
        
        if info and not hist.empty:
            curr_price = hist['Close'].iloc[-1]
            st.metric(f"{info.get('shortName', ticker)}", f"{info.get('currency', '')} {curr_price:,.2f}")
            st.divider()

            if view_mode == "🧠 Strategy":
                # --- STRATEGY ENGINE (V12.0 Logic) ---
                method = st.selectbox("Select Strategy:", ["🚀 Growth Hunter (CAN SLIM)", "🪄 Bargain Hunter (Magic Formula)", "Fortress Method (MOAT)", "🏦 Bankers' Method (CAMELS)", "🏇 Jockey Analysis (Management)", "🕵️ Scuttlebutt (Research)"])
                
                # ... (Strategy Code - Compressed for readability) ...
                if method == "🚀 Growth Hunter (CAN SLIM)":
                    st.markdown("<div class='method-card'><h3>🚀 CAN SLIM Analysis</h3><p>Focus: High Growth.</p></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("EPS Growth", f"{info.get('earningsGrowth', 0)*100:.1f}%")
                    c2.metric("Rev Growth", f"{info.get('revenueGrowth', 0)*100:.1f}%")
                elif method == "🪄 Bargain Hunter (Magic Formula)":
                    st.markdown("<div class='method-card'><h3>🪄 Magic Formula</h3><p>Focus: Quality + Value.</p></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    pe = info.get('trailingPE', 0)
                    c1.metric("Earnings Yield", f"{(1/pe * 100):.2f}%" if pe else "N/A")
                    c2.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
                elif method == "Fortress Method (MOAT)":
                    st.markdown("<div class='method-card'><h3>🏰 MOAT Analysis</h3><p>Focus: Durable Advantage.</p></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Gross Margin", f"{info.get('grossMargins', 0)*100:.1f}%")
                    c2.metric("Debt/Equity", f"{info.get('debtToEquity', 0)}%")
                elif method == "🏦 Bankers' Method (CAMELS)":
                    st.markdown("<div class='method-card'><h3>🏦 CAMEL(S)</h3><p>Focus: Bank Safety.</p></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Debt/Equity", f"{info.get('debtToEquity', 'N/A')}")
                    c2.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.2f}%")
                elif method == "🏇 Jockey Analysis (Management)":
                    st.markdown("<div class='method-card'><h3>🏇 Jockey Analysis</h3><p>Focus: Insider Alignment.</p></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Insider Holding", f"{info.get('heldPercentInsiders', 0)*100:.1f}%")
                    c2.metric("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "0%")
                elif method == "🕵️ Scuttlebutt (Research)":
                    st.markdown("<div class='method-card'><h3>🕵️ Scuttlebutt</h3><p>Soft Data Scan.</p></div>", unsafe_allow_html=True)
                    render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} reviews scandal")]))

                st.divider()
                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                fig.update_layout(height=400, template="plotly_dark", title=f"{ticker} Price Action")
                st.plotly_chart(fig, use_container_width=True)

            elif view_mode == "📑 Financials":
                st.subheader(f"📑 Financials: {info.get('shortName', ticker)} (In Crores)")
                st.info("Values are divided by 1,00,00,000 (1 Cr) for readability.")
                
                f1, f2, f3 = st.tabs(["Income", "Balance Sheet", "Cash Flow"])
                
                # Function to convert to Crores
                def to_crores(df):
                    if df is not None:
                        # Try to convert everything to numeric, errors to NaN
                        df = df.apply(pd.to_numeric, errors='coerce')
                        # Divide by 1 Crore
                        return df.div(10000000)
                    return None

                fin_cr = to_crores(fin)
                bal_cr = to_crores(bal)
                cash_cr = to_crores(cash)

                with f1:
                    if fin_cr is not None: st.dataframe(fin_cr.style.format("{:,.2f} Cr"), use_container_width=True, height=500)
                    else: st.warning("Income Statement Unavailable")
                with f2:
                    if bal_cr is not None: st.dataframe(bal_cr.style.format("{:,.2f} Cr"), use_container_width=True, height=500)
                    else: st.warning("Balance Sheet Unavailable")
                with f3:
                    if cash_cr is not None: st.dataframe(cash_cr.style.format("{:,.2f} Cr"), use_container_width=True, height=500)
                    else: st.warning("Cash Flow Unavailable")