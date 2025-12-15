import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
import json
import os
import concurrent.futures
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar 18.0", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"
TRANSACTION_FILE = "transactions.json"

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
    .metric-delta-pos { color: #00C805; font-weight: 600; font-size: 14px; }
    .metric-delta-neg { color: #FF3B30; font-weight: 600; font-size: 14px; }
    
    /* Strategy Cards */
    .method-card { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #64B5F6; margin-bottom: 20px; }
    .verdict-pass { background-color: rgba(76, 175, 80, 0.1); color: #4CAF50; border: 1px solid #4CAF50; padding: 5px 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .verdict-fail { background-color: rgba(244, 67, 54, 0.1); color: #FF5252; border: 1px solid #FF5252; padding: 5px 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    
    /* News */
    .news-card { border-left: 3px solid #4CAF50; background-color: #262730; padding: 12px; margin-bottom: 10px; border-radius: 6px; transition: 0.3s; }
    .news-card:hover { background-color: #2E303A; }
    .news-title { font-size: 15px; font-weight: 600; color: #E0E0E0; text-decoration: none; }
    
    /* Tables */
    .dataframe { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_json(filename, default):
    if os.path.exists(filename):
        try: with open(filename, 'r') as f: return json.load(f)
        except: return default
    return default

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

if 'watchlist' not in st.session_state: st.session_state.watchlist = load_json(WATCHLIST_FILE, {"india": [], "global": []})
if 'trading' not in st.session_state: st.session_state.trading = load_json(TRADING_FILE, {"india": {"cash": 1000000.0, "holdings": {}}, "global": {"cash": 100000.0, "holdings": {}}})
if 'transactions' not in st.session_state: st.session_state.transactions = load_json(TRANSACTION_FILE, [])

# --- BACKEND FUNCTIONS ---

def get_google_rss(query): 
    return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

def safe_float(val):
    try: return float(val) if val is not None else 0.0
    except: return 0.0

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

@st.cache_data(ttl=300)
def get_market_movers_india():
    """Scans top Indian stocks to find Gainers vs Losers"""
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS"]
    movers = []
    def fetch_change(t):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="2d")
            if len(hist) > 1:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                return {"Symbol": t.replace('.NS', ''), "Price": curr, "Change": chg}
        except: return None
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_change, tickers))
    
    clean_results = [r for r in results if r is not None]
    clean_results.sort(key=lambda x: x['Change'], reverse=True)
    return clean_results

@st.cache_data(ttl=300)
def get_screener_data(tickers):
    def fetch_metrics(t):
        try:
            s = yf.Ticker(t)
            i = s.info
            return {
                "Ticker": t,
                "Price": i.get('currentPrice', 0),
                "P/E": i.get('trailingPE', 0),
                "PEG": i.get('pegRatio', 0),
                "ROE %": i.get('returnOnEquity', 0) * 100 if i.get('returnOnEquity') else 0,
                "Debt/Eq": i.get('debtToEquity', 0),
            }
        except: return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_metrics, tickers))
    return [r for r in results if r]

@st.cache_data(ttl=300)
def get_ticker_data_parallel(tickers):
    def fetch(t):
        try:
            s = yf.Ticker(t)
            h = s.history(period="5d")
            if len(h) > 1:
                return {
                    "symbol": t, "price": h['Close'].iloc[-1], 
                    "change": ((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2])*100,
                    "high": h['High'].iloc[-1], "low": h['Low'].iloc[-1], "hist": h['Close'].tolist()
                }
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
            if hasattr(f, 'bozo') and f.bozo == 1: return []
            return [{
                "title": e.title, "link": e.link, 
                "source": e.source.title if 'source' in e else "News", 
                "date": e.published if 'published' in e else datetime.now().strftime("%Y-%m-%d"),
                "timestamp": e.published_parsed if 'published_parsed' in e else time.localtime()
            } for e in f.entries[:5]]
        except: return []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in executor.map(fetch, url_list): all_news.extend(res)
    
    # Sorting Newest to Oldest
    all_news.sort(key=lambda x: x['timestamp'], reverse=True)
    return all_news[:10]

@st.cache_data(ttl=3600)
def get_deep_company_data(ticker):
    try:
        s = yf.Ticker(ticker)
        return s.info, s.history(period="1y"), s.financials, s.balance_sheet, s.cashflow
    except: return None, None, None, None, None

# --- RENDERERS ---

def render_pro_metrics(data_list):
    if not data_list: st.caption("Loading..."); return
    cols = st.columns(len(data_list)) if len(data_list) <= 4 else st.columns(4)
    for i, d in enumerate(data_list):
        col = cols[i % 4] if i >= 4 else cols[i]
        with col:
            c = "#00C805" if d['change'] >= 0 else "#FF3B30"
            bg = f"rgba({0 if d['change']>=0 else 255}, {200 if d['change']>=0 else 59}, {5 if d['change']>=0 else 48}, 0.1)"
            denom = d['high'] - d['low']
            rng = ((d['price'] - d['low']) / denom) * 100 if denom > 0 else 50
            
            st.markdown(f"""
            <div class="metric-container" style="border-left: 4px solid {c}; background: linear-gradient(180deg, #1E1E1E 0%, {bg} 100%);">
                <div style="font-size:12px; color:#aaa; font-weight:bold;">{d['symbol']}</div>
                <div class="metric-value" style="color:{c}">{d['price']:,.2f}</div>
                <div class="{ 'metric-delta-pos' if d['change']>=0 else 'metric-delta-neg' }">{d['change']:+.2f}%</div>
                <div style="margin-top: 8px; height:4px; background:#333; position:relative;">
                    <div style="width:{rng}%; height:100%; background:{c}; position:absolute;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            fig = go.Figure(data=go.Scatter(y=d['hist'], mode='lines', fill='tozeroy', line=dict(color=c, width=2), fillcolor=bg))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=35, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

def render_news(news):
    if not news: st.caption("No recent updates."); return
    for n in news:
        display_date = n['date'][:16] if len(n['date']) > 16 else n['date']
        st.markdown(f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:10px; color:#4CAF50; font-weight:bold;">{n['source']}</span>
                <span style="font-size:10px; color:#888;">{display_date}</span>
            </div>
            <a href="{n['link']}" class="news-title" target="_blank">{n['title']}</a>
        </div>""", unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("🦅 Executive Market Radar 18.0")
st.caption("Strategic Intelligence | BlackRock Edition")

tab_india, tab_global, tab_ceo, tab_trade, tab_analyst = st.tabs([
    "🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "📈 Trading Floor", "🧠 Analyst Lab"
])

# --- TAB 1: INDIA (Upgraded with Commodities) ---
with tab_india:
    st.markdown("<div class='section-header'>📊 Market Pulse</div>", unsafe_allow_html=True)
    # Indices + Commodities
    tickers = ["^NSEI", "^BSESN", "^NSEBANK", "GC=F", "SI=F", "CL=F"] 
    data = get_ticker_data_parallel(tickers)
    
    # Custom Labels
    label_map = {"GC=F": "GOLD", "SI=F": "SILVER", "CL=F": "CRUDE OIL", "^NSEI": "NIFTY 50", "^BSESN": "SENSEX"}
    if data:
        for d in data: d['symbol'] = label_map.get(d['symbol'], d['symbol'])
        render_pro_metrics(data)

    if st.session_state.watchlist["india"]: 
        st.subheader("⭐ Watchlist")
        render_pro_metrics(get_ticker_data_parallel(st.session_state.watchlist["india"]))
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("**🚀 Growth & Tech**"); render_news(fetch_feed_parallel([get_google_rss("Indian Startup Funding"), get_google_rss("Nifty IT News")]))
    with c2: st.markdown("**🏦 Finance & Policy**"); render_news(fetch_feed_parallel([get_google_rss("RBI Policy India"), get_google_rss("Indian Bank Stocks News")]))
    with c3: st.markdown("**🛢️ Commodities & Infra**"); render_news(fetch_feed_parallel([get_google_rss("India Infrastructure News"), get_google_rss("Gold Price India")]))

# --- TAB 2: GLOBAL (Upgraded with Commodities) ---
with tab_global:
    st.markdown("<div class='section-header'>🌍 Global Pulse</div>", unsafe_allow_html=True)
    tickers = ["^GSPC", "^IXIC", "BTC-USD", "GC=F", "HG=F", "NG=F"]
    data = get_ticker_data_parallel(tickers)
    label_map_gl = {"HG=F": "COPPER", "NG=F": "NATURAL GAS", "^GSPC": "S&P 500", "BTC-USD": "BITCOIN"}
    if data:
        for d in data: d['symbol'] = label_map_gl.get(d['symbol'], d['symbol'])
        render_pro_metrics(data)
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.markdown("**🇺🇸 Wall St & Fed**"); render_news(fetch_feed_parallel([get_google_rss("Federal Reserve News"), get_google_rss("Wall Street Market Analysis")]))
    with c2: st.markdown("**🌏 Geopolitics & Energy**"); render_news(fetch_feed_parallel([get_google_rss("Global Oil Prices OPEC"), get_google_rss("China Economy News")]))

# --- TAB 3: CEO RADAR (V16 Core + V17 Enhancements) ---
with tab_ceo:
    st.markdown("<div class='section-header'>🏛️ Strategic Situation Room</div>", unsafe_allow_html=True)
    
    # 1. Yield Curve (Preserved from 16.0)
    c_yield, c_pulse = st.columns([2, 1])
    with c_yield:
        st.subheader("⚠️ US Yield Curve (Recession Watch)")
        labels, values = get_yield_curve_data()
        if labels:
            fig = go.Figure(go.Scatter(x=labels, y=values, mode='lines+markers', line=dict(color='#FFA726', width=4)))
            fig.update_layout(height=250, title="Yield Curve", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Yield %")
            st.plotly_chart(fig, use_container_width=True)
    
    # 2. Economic Pulse (Split India/Global - V17 Request)
    with c_pulse:
        st.subheader("🏗️ Economic Pulse")
        rd = get_ticker_data_parallel(["USDINR=X", "DX-Y.NYB", "^TNX"])
        pmap = {d['symbol']: d['price'] for d in rd}
        
        t1, t2 = st.tabs(["🇮🇳 India", "🌎 Global"])
        with t1:
            if "USDINR=X" in pmap: st.metric("USD/INR", f"₹{pmap['USDINR=X']:.2f}")
        with t2:
            if "DX-Y.NYB" in pmap: st.metric("Dollar Index", f"{pmap['DX-Y.NYB']:.2f}")
            if "^TNX" in pmap: st.metric("US 10Y Yield", f"{pmap['^TNX']:.2f}%")

    st.divider()
    
    # 3. Market Movers Engine (Added from V17)
    st.subheader("🏆 Market Movers (Nifty Leaders)")
    movers = get_market_movers_india()
    if movers:
        g, l = st.columns(2)
        gainers = [m for m in movers if m['Change'] > 0][:5]
        losers = [m for m in movers if m['Change'] < 0][:5]
        with g:
            st.success("Top Gainers")
            for m in gainers: st.markdown(f"**{m['Symbol']}**: {m['Change']:.2f}%")
        with l:
            st.error("Top Losers")
            for m in losers: st.markdown(f"**{m['Symbol']}**: {m['Change']:.2f}%")
            
    st.divider()
    
    # 4. Sector Heatmaps (Preserved from V16)
    st.subheader("🔥 Sector Performance")
    def plot_treemap(sector_dict, title):
        sd = get_ticker_data_parallel(list(sector_dict.values()))
        if sd:
            df = pd.DataFrame([{"Sector": k, "Change": next((x['change'] for x in sd if x['symbol'] == v), 0)} for k, v in sector_dict.items()])
            fig = px.treemap(df, path=['Sector'], values=[10]*len(df), color='Change', color_continuous_scale=['#FF5252', '#222', '#4CAF50'], range_color=[-2, 2])
            fig.update_layout(height=300, margin=dict(t=30,b=0,l=0,r=0), title=title)
            return fig
        return None
    
    h1, h2 = st.columns(2)
    with h1: 
        f = plot_treemap({"Banks": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Energy": "^CNXENERGY"}, "India Sectors")
        if f: st.plotly_chart(f, use_container_width=True)
    with h2:
        f = plot_treemap({"Tech": "IXN", "Energy": "IXC", "Finance": "IXG"}, "Global Sectors")
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 4: TRADING FLOOR (Smart Search Fixed) ---
with tab_trade:
    if 'trading' in st.session_state:
        st.markdown("<div class='section-header'>📈 Virtual Exchange</div>", unsafe_allow_html=True)
        mkt = st.radio("Market", ["🇮🇳 India", "🇺🇸 Global"], horizontal=True)
        m_key = "india" if "India" in mkt else "global"
        curr = "₹" if "India" in mkt else "$"
        port = st.session_state.trading[m_key]
        
        c1, c2 = st.columns(2)
        c1.metric("Cash", f"{curr}{port['cash']:,.0f}")
        
        # Smart Search
        t_sym = st.text_input("Trade Ticker (e.g., ZOMATO)", "RELIANCE").upper()
        
        # Auto-append .NS if missing for India
        if m_key == "india" and not t_sym.endswith(".NS") and len(t_sym) > 0:
            final_ticker = f"{t_sym}.NS"
        else:
            final_ticker = t_sym

        if final_ticker:
            d = get_ticker_data_parallel([final_ticker])
            if d:
                lp = d[0]['price']
                st.success(f"Verified: {d[0]['symbol']} @ {curr}{lp:,.2f}")
                
                c_act, c_qty, c_btn = st.columns([1,1,1])
                act = c_act.selectbox("Action", ["BUY", "SELL"])
                qty = c_qty.number_input("Qty", 1, 10000)
                
                if c_btn.button("EXECUTE"):
                    val = lp * qty
                    fee = val * 0.001
                    if act == "BUY":
                        if port['cash'] >= (val+fee):
                            port['cash'] -= (val+fee)
                            if final_ticker in port['holdings']:
                                old = port['holdings'][final_ticker]
                                new_q = old['qty'] + qty
                                new_avg = ((old['qty']*old['avg_price'])+val)/new_q
                                port['holdings'][final_ticker] = {'qty': new_q, 'avg_price': new_avg}
                            else:
                                port['holdings'][final_ticker] = {'qty': qty, 'avg_price': lp}
                            st.success("Bought!")
                            save_json(TRADING_FILE, st.session_state.trading)
                            st.rerun()
                        else: st.error("No Funds")
                    elif act == "SELL":
                        if final_ticker in port['holdings'] and port['holdings'][final_ticker]['qty'] >= qty:
                            port['cash'] += (val-fee)
                            port['holdings'][final_ticker]['qty'] -= qty
                            if port['holdings'][final_ticker]['qty'] == 0: del port['holdings'][final_ticker]
                            st.success("Sold!")
                            save_json(TRADING_FILE, st.session_state.trading)
                            st.rerun()
                        else: st.error("No Shares")
            elif t_sym: st.caption("Searching...")

        st.subheader("Holdings")
        if port['holdings']:
            st.dataframe(pd.DataFrame([{"Ticker": k, "Qty": v['qty'], "Avg": f"{curr}{v['avg_price']:.2f}"} for k,v in port['holdings'].items()]), use_container_width=True)

# --- TAB 5: ANALYST LAB (V16 Core + Scuttlebutt Upgrade) ---
with tab_analyst:
    st.markdown("<div class='section-header'>🔍 Analyst Masterclass</div>", unsafe_allow_html=True)
    mode = st.radio("Mode:", ["🧠 Deep Dive", "⚡ Screener"], horizontal=True)
    
    if mode == "⚡ Screener":
        st.subheader("⚡ Live Screener (Watchlist + Nifty)")
        if st.button("Run Scan"):
            scan_list = list(set(st.session_state.watchlist["india"] + ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]))
            res = get_screener_data(scan_list)
            if res: st.dataframe(pd.DataFrame(res).style.format({"Price": "{:.2f}"}), use_container_width=True)
            
    else:
        c_in, c_view = st.columns([2, 1])
        with c_in: ticker = st.text_input("Analyze Ticker:", "RELIANCE.NS").upper()
        with c_view: view_type = st.selectbox("View:", ["Strategy Scorecards", "Deep Financials"])
        
        if ticker:
            info, hist, fin, bal, cash = get_deep_company_data(ticker)
            if info and not hist.empty:
                st.metric(info.get('shortName', ticker), f"{hist['Close'].iloc[-1]:.2f}")
                
                # CHART (Restored from V16)
                st.subheader("Price Action")
                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.divider()

                if view_type == "Strategy Scorecards":
                    strat = st.selectbox("Framework:", ["🚀 CAN SLIM", "🪄 Magic Formula", "🏰 MOAT", "🏦 CAMELS (Bank)", "🏇 Jockey (Mgmt)", "🕵️ Scuttlebutt"])
                    
                    def get_val(k, d=0): return safe_float(info.get(k, d))

                    # (V16 Strategy Logic Preserved)
                    if strat == "🚀 CAN SLIM":
                        c1, c2 = st.columns(2)
                        eps = get_val('earningsGrowth')
                        c1.metric("EPS Growth", f"{eps*100:.1f}%", delta="Target > 20%")
                        if eps > 0.20: st.markdown("<div class='verdict-pass'>PASS</div>", unsafe_allow_html=True)
                        else: st.markdown("<div class='verdict-fail'>FAIL</div>", unsafe_allow_html=True)
                    
                    elif strat == "🏰 MOAT":
                        c1, c2 = st.columns(2)
                        roe = get_val('returnOnEquity')
                        c1.metric("ROE", f"{roe*100:.1f}%", delta="Target > 15%")
                        if roe > 0.15: st.markdown("<div class='verdict-pass'>WIDE MOAT</div>", unsafe_allow_html=True)
                    
                    elif strat == "🕵️ Scuttlebutt":
                        st.markdown("<div class='method-card'><h3>🕵️ Scuttlebutt Intel (Upgraded)</h3></div>", unsafe_allow_html=True)
                        c1, c2, c3 = st.columns(3)
                        with c1: 
                            st.markdown("**⚖️ Legal & Governance**")
                            render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} fraud lawsuit")]))
                        with c2:
                            st.markdown("**👔 Management**")
                            render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} CEO interview")]))
                        with c3:
                            st.markdown("**📦 Product & Brand**")
                            render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} reviews complaints")]))

                    # "Why is it moving?" (Added feature)
                    st.divider()
                    st.subheader("📉 Why is the stock moving?")
                    render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} share price reason analysis")]))

                elif view_type == "Deep Financials":
                    st.subheader("📑 Statements (In Crores)")
                    def to_cr(df): return df.div(10000000) if df is not None else None
                    t1, t2 = st.tabs(["Income", "Balance"])
                    with t1: st.dataframe(to_cr(fin).style.format("{:,.2f} Cr") if fin is not None else None, use_container_width=True)
                    with t2: st.dataframe(to_cr(bal).style.format("{:,.2f} Cr") if bal is not None else None, use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Watchlist")
    it = st.text_input("Add Stock", key="wb").upper()
    if st.button("Add"): 
        st.session_state.watchlist["india"].append(it); save_json(WATCHLIST_FILE, st.session_state.watchlist); st.rerun()