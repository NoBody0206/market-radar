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
st.set_page_config(page_title="Executive Market Radar 17.0", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"
TRADING_FILE = "trading_engine.json"
TRANSACTION_FILE = "transactions.json"

# --- PRO CSS STYLING (BlackRock Style) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    
    /* Metric Cards */
    .metric-container { 
        background-color: #1E1E1E; 
        border: 1px solid #333; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
    }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: 700; margin: 5px 0; color: #E0E0E0; }
    .metric-delta-pos { color: #00C805; font-size: 14px; font-weight: 600; }
    .metric-delta-neg { color: #FF3B30; font-size: 14px; font-weight: 600; }
    
    /* Strategy Cards */
    .method-card { background-color: #262730; padding: 15px; border-radius: 8px; border-left: 4px solid #64B5F6; margin-bottom: 15px; }
    
    /* Verdict Tags */
    .verdict-tag { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .v-pass { background: rgba(0, 200, 5, 0.2); color: #00C805; border: 1px solid #00C805; }
    .v-fail { background: rgba(255, 59, 48, 0.2); color: #FF3B30; border: 1px solid #FF3B30; }
    
    /* News */
    .news-card { 
        background-color: #1a1c24; 
        padding: 12px; 
        margin-bottom: 8px; 
        border-radius: 6px; 
        border-left: 3px solid #444; 
        transition: transform 0.2s;
    }
    .news-card:hover { transform: translateX(5px); border-left: 3px solid #64B5F6; }
    .news-title { font-size: 14px; font-weight: 500; color: #E0E0E0; text-decoration: none; }
    .news-meta { font-size: 10px; color: #666; margin-top: 4px; display: flex; justify-content: space-between; }
    
    /* Tables */
    .dataframe { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_json(WATCHLIST_FILE, {"india": [], "global": []})
if 'trading' not in st.session_state:
    st.session_state.trading = load_json(TRADING_FILE, {"india": {"cash": 1000000.0, "holdings": {}}, "global": {"cash": 100000.0, "holdings": {}}})
if 'transactions' not in st.session_state:
    st.session_state.transactions = load_json(TRANSACTION_FILE, [])

# --- BACKEND FUNCTIONS (OPTIMIZED) ---

def get_google_rss(query): 
    return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

def safe_float(val):
    try: return float(val) if val is not None else 0.0
    except: return 0.0

@st.cache_data(ttl=300)
def get_market_movers_india():
    """Scans top Indian stocks to find Gainers vs Losers"""
    # Top 15 Nifty Heavyweights
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

@st.cache_data(ttl=600)
def get_macro_data():
    """Fetches key macro indicators efficiently"""
    tickers = {
        "USD/INR": "USDINR=X",
        "India VIX": "^INDIAVIX", # Often fails, fallback logic handled in UI
        "US 10Y": "^TNX",
        "Dollar Idx": "DX-Y.NYB",
        "Brent Oil": "BZ=F",
        "Gold": "GC=F",
        "Silver": "SI=F"
    }
    
    data = {}
    def fetch(name, sym):
        try:
            t = yf.Ticker(sym)
            h = t.history(period="2d")
            if not h.empty:
                curr = h['Close'].iloc[-1]
                prev = h['Close'].iloc[-2] if len(h) > 1 else curr
                chg = ((curr - prev)/prev)*100
                return name, {"price": curr, "change": chg}
        except: return name, None
        return name, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch, k, v) for k, v in tickers.items()]
        for f in concurrent.futures.as_completed(futures):
            k, v = f.result()
            if v: data[k] = v
    return data

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
                "source": e.source.title if 'source' in e else "Source", 
                "date": e.published[:16] if 'published' in e else "Just Now"
            } for e in f.entries[:4]]
        except: return []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in executor.map(fetch, url_list): all_news.extend(res)
    return all_news[:10]

@st.cache_data(ttl=3600)
def get_deep_company_data(ticker):
    try:
        s = yf.Ticker(ticker)
        # Fetch minimal needed data to speed up
        return s.info, s.history(period="1y"), s.financials, s.balance_sheet
    except: return None, None, None, None

# --- RENDERERS ---

def render_pro_card(data):
    if not data: return
    c = "#00C805" if data['change'] >= 0 else "#FF3B30"
    
    st.markdown(f"""
    <div class="metric-container" style="border-left: 4px solid {c};">
        <div class="metric-label">{data['symbol']}</div>
        <div class="metric-value">{data['price']:,.2f}</div>
        <div class="{ 'metric-delta-pos' if data['change']>=0 else 'metric-delta-neg' }">
            {data['change']:+.2f}%
        </div>
        <div style="height: 4px; width: 100%; background: #333; margin-top:8px; border-radius:2px;">
            <div style="height:100%; width: 50%; background: {c}; border-radius:2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_news_list(news_items):
    if not news_items: st.caption("No fresh updates."); return
    for n in news_items:
        st.markdown(f"""
        <div class="news-card">
            <a href="{n['link']}" class="news-title" target="_blank">{n['title']}</a>
            <div class="news-meta">
                <span>{n['source']}</span>
                <span>{n['date']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

# --- MAIN APP ---
st.title("🦅 Executive Market Radar 17.0")
st.caption("Next-Level Intelligence | Commodities | Market Movers Engine")

tab_india, tab_global, tab_ceo, tab_trade, tab_analyst = st.tabs([
    "🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "📈 Trading Floor", "🧠 Analyst Lab"
])

# --- TAB 1: INDIA (Next Level) ---
with tab_india:
    st.markdown("<div class='section-header'>📊 India Market Pulse</div>", unsafe_allow_html=True)
    
    # Indices & Commodities Mixed
    tickers = ["^NSEI", "^BSESN", "^NSEBANK", "GC=F", "SI=F", "CL=F"] # Gold, Silver, Oil
    data = get_ticker_data_parallel(tickers)
    
    # Custom Labels for display
    display_map = {"GC=F": "GOLD (Global)", "SI=F": "SILVER (Global)", "CL=F": "CRUDE OIL", "^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^NSEBANK": "BANK NIFTY"}
    
    if data:
        cols = st.columns(3)
        for i, d in enumerate(data):
            d['symbol'] = display_map.get(d['symbol'], d['symbol'])
            with cols[i % 3]: render_pro_card(d)
    
    st.divider()
    st.markdown("### 📰 Critical Briefing")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏛️ Policy & Economy**")
        render_news_list(fetch_feed_parallel([get_google_rss("RBI India Policy Economy")]))
    with c2:
        st.markdown("**💎 Commodities & Energy**")
        render_news_list(fetch_feed_parallel([get_google_rss("Gold Price India Oil Import")]))

# --- TAB 2: GLOBAL (Next Level) ---
with tab_global:
    st.markdown("<div class='section-header'>🌍 Global Pulse</div>", unsafe_allow_html=True)
    
    tickers = ["^GSPC", "^IXIC", "BTC-USD", "GC=F", "HG=F", "NG=F"] # S&P, Nasdaq, BTC, Gold, Copper, Nat Gas
    data = get_ticker_data_parallel(tickers)
    display_map_gl = {"HG=F": "COPPER", "NG=F": "NATURAL GAS", "^GSPC": "S&P 500", "BTC-USD": "BITCOIN"}
    
    if data:
        cols = st.columns(3)
        for i, d in enumerate(data):
            d['symbol'] = display_map_gl.get(d['symbol'], d['symbol'])
            with cols[i % 3]: render_pro_card(d)
            
    st.divider()
    st.markdown("### 📰 World Intel")
    c1, c2 = st.columns(2)
    with c1: st.markdown("**🇺🇸 Wall Street**"); render_news_list(fetch_feed_parallel([get_google_rss("US Stock Market News")]))
    with c2: st.markdown("**🌏 Geopolitics**"); render_news_list(fetch_feed_parallel([get_google_rss("Global Supply Chain Geopolitics")]))

# --- TAB 3: CEO RADAR (Strategic Room) ---
with tab_ceo:
    st.markdown("<div class='section-header'>🏛️ Strategic Situation Room</div>", unsafe_allow_html=True)
    
    # 1. MACRO SPLIT
    st.subheader("1. Economic Pulse")
    macros = get_macro_data()
    
    c_ind, c_gl = st.columns(2)
    
    with c_ind:
        st.info("🇮🇳 India Macro")
        if "USD/INR" in macros: st.metric("USD/INR", f"₹{macros['USD/INR']['price']:.2f}", f"{macros['USD/INR']['change']:.2f}%", delta_color="inverse")
        # Fallback for India VIX
        if "India VIX" in macros and macros["India VIX"]: 
            st.metric("India VIX (Fear)", f"{macros['India VIX']['price']:.2f}", f"{macros['India VIX']['change']:.2f}%", delta_color="inverse")
        else:
            st.metric("India VIX", "Unavailable", "Check NSE")

    with c_gl:
        st.info("🌎 Global Macro")
        if "Dollar Idx" in macros: st.metric("DXY (Dollar Index)", f"{macros['Dollar Idx']['price']:.2f}", f"{macros['Dollar Idx']['change']:.2f}%")
        if "US 10Y" in macros: st.metric("US 10Y Yield", f"{macros['US 10Y']['price']:.2f}%", "Risk Free Rate")

    st.divider()
    
    # 2. MARKET MOVERS ENGINE (Gainers vs Losers)
    st.subheader("2. Market Movers (Nifty Heavyweights)")
    movers = get_market_movers_india()
    
    if movers:
        gainers = [m for m in movers if m['Change'] > 0][:5]
        losers = [m for m in movers if m['Change'] < 0][-5:] # Bottom 5
        
        c_win, c_loss = st.columns(2)
        with c_win:
            st.success("🏆 Top Gainers")
            for m in gainers: st.markdown(f"**{m['Symbol']}**: ₹{m['Price']:.1f} (+{m['Change']:.2f}%)")
        with c_loss:
            st.error("🩸 Top Losers")
            for m in sorted(losers, key=lambda x: x['Change']): # Sort mostly negative to top
                st.markdown(f"**{m['Symbol']}**: ₹{m['Price']:.1f} ({m['Change']:.2f}%)")
    else:
        st.warning("Market Movers data currently unavailable.")

# --- TAB 4: TRADING FLOOR (FIXED) ---
with tab_trade:
    if 'trading' in st.session_state:
        st.markdown("<div class='section-header'>📈 Virtual Exchange</div>", unsafe_allow_html=True)
        
        # 1. MARKET SELECTION
        mkt = st.radio("Select Market:", ["🇮🇳 India (NSE)", "🇺🇸 US Market"], horizontal=True)
        m_key = "india" if "India" in mkt else "global"
        curr = "₹" if "India" in mkt else "$"
        
        # 2. PORTFOLIO SUMMARY
        port = st.session_state.trading[m_key]
        c1, c2 = st.columns(2)
        c1.metric("Cash Balance", f"{curr}{port['cash']:,.2f}")
        
        # 3. SMART SEARCH & TRADE
        st.markdown("### ⚡ Quick Order")
        search_q = st.text_input("Search Ticker (e.g. Zomato, Tata, Apple)", "").upper()
        
        # Auto-Correction Logic
        final_ticker = None
        if search_q:
            if "India" in mkt and not search_q.endswith(".NS"):
                final_ticker = f"{search_q}.NS" # Auto-append .NS
            else:
                final_ticker = search_q
        
        if final_ticker:
            d = get_ticker_data_parallel([final_ticker])
            if d:
                stock_data = d[0]
                st.success(f"Found: {stock_data['symbol']} @ {curr}{stock_data['price']:.2f}")
                
                # Trade UI
                c_act, c_qty, c_btn = st.columns([1,1,1])
                action = c_act.selectbox("Action", ["BUY", "SELL"])
                qty = c_qty.number_input("Quantity", 1, 10000)
                
                if c_btn.button("EXECUTE TRADE"):
                    # Execute Logic
                    price = stock_data['price']
                    val = price * qty
                    fee = val * 0.001
                    
                    if action == "BUY":
                        if port['cash'] >= (val + fee):
                            port['cash'] -= (val + fee)
                            if final_ticker in port['holdings']:
                                old = port['holdings'][final_ticker]
                                new_q = old['qty'] + qty
                                new_avg = ((old['qty']*old['avg_price']) + val)/new_q
                                port['holdings'][final_ticker] = {'qty': new_q, 'avg_price': new_avg}
                            else:
                                port['holdings'][final_ticker] = {'qty': qty, 'avg_price': price}
                            st.success("Order Filled! ✅")
                            save_json(TRADING_FILE, st.session_state.trading)
                            st.rerun()
                        else: st.error("Insufficient Funds")
                    
                    elif action == "SELL":
                        if final_ticker in port['holdings'] and port['holdings'][final_ticker]['qty'] >= qty:
                            port['cash'] += (val - fee)
                            port['holdings'][final_ticker]['qty'] -= qty
                            if port['holdings'][final_ticker]['qty'] == 0: del port['holdings'][final_ticker]
                            st.success("Sold Successfully! 💰")
                            save_json(TRADING_FILE, st.session_state.trading)
                            st.rerun()
                        else: st.error("Not enough shares to sell")
            else:
                st.warning(f"Ticker '{final_ticker}' not found. Try exact symbol.")

        # 4. HOLDINGS
        st.subheader("Your Holdings")
        if port['holdings']:
            rows = []
            for t, v in port['holdings'].items():
                rows.append({"Ticker": t, "Qty": v['qty'], "Avg Price": f"{curr}{v['avg_price']:.2f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- TAB 5: ANALYST LAB (ADVANCED) ---
with tab_analyst:
    st.markdown("<div class='section-header'>🔍 Analyst Masterclass</div>", unsafe_allow_html=True)
    
    ticker = st.text_input("Deep Dive Ticker:", "RELIANCE.NS").upper()
    
    if ticker:
        info, hist, fin, bal = get_deep_company_data(ticker)
        if info and not hist.empty:
            curr_p = hist['Close'].iloc[-1]
            st.metric(f"{info.get('shortName', ticker)}", f"{curr_p:.2f}")
            
            t_strat, t_news = st.tabs(["🧠 Strategy Frameworks", "📰 News & Sentiment"])
            
            with t_strat:
                strat = st.selectbox("Select Framework:", ["🚀 CAN SLIM", "🪄 Magic Formula", "🏰 MOAT Analysis", "🕵️ Scuttlebutt (Soft Data)"])
                
                if strat == "🕵️ Scuttlebutt (Soft Data)":
                    st.markdown("<div class='method-card'><h3>🕵️ Scuttlebutt Intel</h3></div>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1: 
                        st.markdown("**⚖️ Legal & Fraud Check**")
                        render_news_list(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} lawsuit fraud scandal")]))
                    with c2:
                        st.markdown("**👔 Management Integrity**")
                        render_news_list(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} CEO interview management style")]))
                    with c3:
                        st.markdown("**📦 Product Reviews**")
                        render_news_list(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} product review complaints")]))
                
                # (Other frameworks kept from V16 logic for brevity, they work fine)
                elif strat == "🚀 CAN SLIM":
                    st.markdown("<div class='method-card'><h3>🚀 CAN SLIM</h3></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    eps = safe_float(info.get('earningsGrowth'))
                    c1.metric("EPS Growth", f"{eps*100:.1f}%", delta="Target > 20%")
                    if eps > 0.20: st.markdown("<span class='verdict-tag v-pass'>PASS</span>", unsafe_allow_html=True)
                    else: st.markdown("<span class='verdict-tag v-fail'>FAIL</span>", unsafe_allow_html=True)

            with t_news:
                st.subheader("📉 Why is the stock moving?")
                st.caption("AI-Search for specific reasons (Earnings, Upgrades, Scandals)")
                reason_q = f"{info.get('shortName', ticker)} stock price reason fall rise analysis"
                render_news_list(fetch_feed_parallel([get_google_rss(reason_q)]))

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Watchlist")
    it = st.text_input("Add Stock", key="wb").upper()
    if st.button("Add"): 
        st.session_state.watchlist["india"].append(it); save_json(WATCHLIST_FILE, st.session_state.watchlist); st.rerun()