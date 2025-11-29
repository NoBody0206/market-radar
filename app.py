import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime
import math
import json
import os
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar 6.0", layout="wide", page_icon="🦅")
WATCHLIST_FILE = "watchlist_data.json"

# --- PRO CSS STYLING ---
st.markdown("""
<style>
    /* Optimized Card Design */
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px; /* Reduced padding for compactness */
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 13px; color: #aaa; font-weight: 500; }
    .metric-value { font-size: 24px; font-weight: bold; margin: 2px 0; }
    .metric-delta { font-size: 13px; font-weight: 600; margin-bottom: 5px; }
    
    /* Range Bar Styles */
    .range-container { width: 100%; height: 4px; background-color: #333; border-radius: 2px; margin-top: 8px; position: relative; }
    .range-fill { height: 100%; border-radius: 2px; position: absolute; top: 0; left: 0; }
    .range-marker { width: 2px; height: 10px; background-color: white; position: absolute; top: -3px; box-shadow: 0 0 4px rgba(255,255,255,0.8); }
    .range-labels { display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px; }
    
    /* Badges */
    .badge-good { background: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 2px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #4CAF50; }
    .badge-bad { background: rgba(244, 67, 54, 0.2); color: #F44336; padding: 2px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #F44336; }
    .badge-neutral { background: rgba(255, 193, 7, 0.2); color: #FFC107; padding: 2px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #FFC107; }

    /* Headers */
    .section-header { font-size: 20px; font-weight: 700; margin-top: 20px; margin-bottom: 15px; color: #64B5F6; border-bottom: 1px solid #444; padding-bottom: 5px; }
    .sub-header { font-size: 15px; font-weight: bold; color: #81C784; margin-top: 10px; }
    .news-card { border-left: 3px solid #4CAF50; background-color: #262730; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .news-title { font-size: 14px; font-weight: 600; color: #E0E0E0; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# --- WATCHLIST MANAGEMENT ---
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f: return json.load(f)
    return {"india": [], "global": []}

def save_watchlist(data):
    with open(WATCHLIST_FILE, 'w') as f: json.dump(data, f)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# --- HIGH PERFORMANCE BACKEND (MULTITHREADED) ---

def get_google_rss(query):
    return f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

def fetch_single_ticker(t):
    """Helper function to fetch a single ticker (used in threading)"""
    try:
        stock = yf.Ticker(t)
        hist = stock.history(period="5d") 
        if len(hist) > 1:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            chg = ((curr - prev) / prev) * 100
            day_high = hist['High'].iloc[-1]
            day_low = hist['Low'].iloc[-1]
            return {
                "symbol": t, "price": curr, "change": chg, 
                "high": day_high, "low": day_low,
                "hist": hist['Close'].tolist()
            }
    except: return None

@st.cache_data(ttl=300)
def get_ticker_data_parallel(tickers):
    """Fetches multiple tickers in parallel threads"""
    results = []
    # Use ThreadPool to fetch 10 stocks at once instead of 1 by 1
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return results

def fetch_single_feed(url):
    """Helper for parallel news fetch"""
    try:
        feed = feedparser.parse(url)
        if hasattr(feed, 'bozo') and feed.bozo == 1: return []
        items = []
        for entry in feed.entries[:3]: # Limit to top 3 per feed for speed
            blob = TextBlob(entry.title)
            pol = blob.sentiment.polarity
            color = "🟢" if pol > 0.1 else "🔴" if pol < -0.1 else "⚪"
            date_str = entry.published[:17] if 'published' in entry else "Recent"
            items.append({"title": entry.title, "link": entry.link, "source": entry.source.title if 'source' in entry else "News", "date": date_str, "mood": color})
        return items
    except: return []

@st.cache_data(ttl=600)
def fetch_feed_parallel(url_list):
    """Fetches RSS feeds in parallel"""
    all_news = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_single_feed, url): url for url in url_list}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: all_news.extend(res)
    return all_news[:10] # Return mixed top 10 results

# --- ANALYST LAB CACHE (HEAVY DATA) ---
@st.cache_data(ttl=3600) # Cache company info for 1 hour (doesn't change often)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.history(period="1y")
    except: return None, None

# --- VISUAL RENDERERS (OPTIMIZED) ---

def render_pro_metrics(data_list):
    if not data_list: 
        st.caption("No data available.")
        return
    
    # Sort data by symbol to keep UI stable
    data_list = sorted(data_list, key=lambda x: x['symbol'])
    
    cols = st.columns(len(data_list)) if len(data_list) <= 4 else st.columns(4)
    for i, d in enumerate(data_list):
        col = cols[i % 4] if i >= 4 else cols[i]
        with col:
            color_code = "#00C805" if d['change'] >= 0 else "#FF3B30"
            bg_tint = "rgba(0, 200, 5, 0.1)" if d['change'] >= 0 else "rgba(255, 59, 48, 0.1)"
            arrow = "▲" if d['change'] >= 0 else "▼"
            
            # Safe division for range
            denom = d['high'] - d['low']
            range_pos = ((d['price'] - d['low']) / denom) * 100 if denom > 0 else 50

            st.markdown(f"""
            <div class="metric-container" style="border-left: 4px solid {color_code}; background: linear-gradient(180deg, #1E1E1E 0%, {bg_tint} 100%);">
                <div class="metric-label">{d['symbol']}</div>
                <div class="metric-value" style="color: {color_code}">{d['price']:,.2f}</div>
                <div class="metric-delta" style="color: {color_code}">{arrow} {d['change']:.2f}%</div>
                <div style="margin-top: 8px;">
                    <div class="range-container">
                        <div class="range-fill" style="width: {range_pos}%; background-color: {color_code}; opacity: 0.5;"></div>
                        <div class="range-marker" style="left: {range_pos}%;"></div>
                    </div>
                    <div class="range-labels"><span>L: {d['low']:,.0f}</span><span>H: {d['high']:,.0f}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            # OPTIMIZED GRAPH (Static Plot)
            fig = go.Figure(data=go.Scatter(y=d['hist'], mode='lines', fill='tozeroy', line=dict(color=color_code, width=2), fillcolor=bg_tint))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=35, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            # config={'staticPlot': True} makes it render much faster (no hover/zoom)
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

def render_news(news_items):
    if not news_items: st.caption("No recent news."); return
    for n in news_items:
        st.markdown(f"""<div class="news-card"><a href="{n['link']}" class="news-title" target="_blank">{n['title']}</a><div class="news-meta"><span style="font-weight:bold; color:#888;">{n['mood']} {n['source']}</span><span>🕒 {n['date']}</span></div></div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Watchlist")
    
    with st.expander("🇮🇳 India", expanded=False):
        in_ticker = st.text_input("Add (e.g. TCS.NS)", key="in_input").upper()
        if st.button("Add IN"):
            if in_ticker and in_ticker not in st.session_state.watchlist["india"]:
                st.session_state.watchlist["india"].append(in_ticker)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        if st.session_state.watchlist["india"]:
            rem_in = st.selectbox("Del IN", st.session_state.watchlist["india"], key="rem_in")
            if st.button("Remove IN"):
                st.session_state.watchlist["india"].remove(rem_in)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

    with st.expander("🌎 Global", expanded=False):
        gl_ticker = st.text_input("Add (e.g. AAPL)", key="gl_input").upper()
        if st.button("Add GL"):
            if gl_ticker and gl_ticker not in st.session_state.watchlist["global"]:
                st.session_state.watchlist["global"].append(gl_ticker)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        if st.session_state.watchlist["global"]:
            rem_gl = st.selectbox("Del GL", st.session_state.watchlist["global"], key="rem_gl")
            if st.button("Remove GL"):
                st.session_state.watchlist["global"].remove(rem_gl)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

    if st.button("🔄 Clear Cache (Fix Data)"):
        st.cache_data.clear()
        st.rerun()

# --- MAIN APP ---
st.title("🦅 Executive Market Radar 6.0")
st.caption("High-Performance Intelligence System | Multithreaded Engine")

tab_india, tab_global, tab_ceo, tab_deep = st.tabs(["🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "🧠 Analyst"])

# --- TAB 1: INDIA ---
with tab_india:
    st.markdown("<div class='section-header'>📊 Benchmarks</div>", unsafe_allow_html=True)
    render_pro_metrics(get_ticker_data_parallel(["^NSEI", "^BSESN", "^NSEBANK", "USDINR=X"]))
    
    if st.session_state.watchlist["india"]:
        st.markdown("<div class='sub-header'>⭐ Watchlist</div>", unsafe_allow_html=True)
        render_pro_metrics(get_ticker_data_parallel(st.session_state.watchlist["india"]))
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='sub-header'>Startups</div>", unsafe_allow_html=True)
        render_news(fetch_feed_parallel([get_google_rss("Indian Startup Funding"), get_google_rss("Venture Capital India")]))
    with c2:
        st.markdown("<div class='sub-header'>Banking</div>", unsafe_allow_html=True)
        render_news(fetch_feed_parallel([get_google_rss("RBI Policy"), get_google_rss("Banking Sector India")]))
    with c3:
        st.markdown("<div class='sub-header'>Corporate</div>", unsafe_allow_html=True)
        render_news(fetch_feed_parallel(["https://www.moneycontrol.com/rss/business.xml"]))

# --- TAB 2: GLOBAL ---
with tab_global:
    st.markdown("<div class='section-header'>🌍 Drivers</div>", unsafe_allow_html=True)
    render_pro_metrics(get_ticker_data_parallel(["^GSPC", "^IXIC", "BTC-USD", "GC=F", "CL=F"]))

    if st.session_state.watchlist["global"]:
        st.markdown("<div class='sub-header'>⭐ Watchlist</div>", unsafe_allow_html=True)
        render_pro_metrics(get_ticker_data_parallel(st.session_state.watchlist["global"]))

    st.divider()
    st.markdown("<div class='sub-header'>Headlines</div>", unsafe_allow_html=True)
    render_news(fetch_feed_parallel(["https://www.cnbc.com/id/100003114/device/rss/rss.html"]))

# --- TAB 3: CEO RADAR ---
with tab_ceo:
    st.markdown("<div class='section-header'>🏛️ Strategic Risk</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yield Curve")
        data = get_ticker_data_parallel(["^TNX"])
        if data:
            tnx = data[0]['price']
            st.metric("10Y Yield", f"{tnx:.2f}%")
            if tnx < 3.5: st.success("Stable 🟢")
            elif tnx > 4.5: st.error("High Risk 🔴")
            else: st.warning("Neutral 🟡")

    with col2:
        st.subheader("Copper/Gold Ratio")
        raw_data = get_ticker_data_parallel(["HG=F", "GC=F"])
        # Map symbol to price
        prices = {d['symbol']: d['price'] for d in raw_data}
        if "HG=F" in prices and "GC=F" in prices:
            ratio = (prices["HG=F"] / prices["GC=F"]) * 1000 
            st.metric("Pulse Score", f"{ratio:.2f}")
            if ratio > 2.0: st.success("Expansive 🚀")
            else: st.error("Defensive 🛡️")
    
    st.divider()
    st.markdown("<div class='sub-header'>Smart Money Reports</div>", unsafe_allow_html=True)
    bank_query = 'site:goldmansachs.com OR site:jpmorgan.com OR site:morganstanley.com "Outlook"'
    render_news(fetch_feed_parallel([get_google_rss(bank_query)]))

# --- TAB 4: ANALYST LAB ---
with tab_deep:
    st.markdown("<div class='section-header'>🔍 Fundamental Scanner</div>", unsafe_allow_html=True)
    ticker = st.text_input("Ticker:", "RELIANCE.NS").upper()

    if ticker:
        info, hist = get_company_info(ticker)
        
        if info and not hist.empty:
            curr = hist['Close'].iloc[-1]
            st.subheader("❤️ Health Card")
            h1, h2, h3, h4 = st.columns(4)
            
            roe = info.get('returnOnEquity', 0)
            h1.markdown(f"**ROE**<br><span class='{'badge-good' if roe > 0.15 else 'badge-bad'}'>{roe*100:.1f}%</span>", unsafe_allow_html=True)

            de = info.get('debtToEquity', 0)
            # Handle cases where debt is None
            de_val = de if de is not None else 0
            h2.markdown(f"**Debt/Eq**<br><span class='{'badge-good' if de_val < 50 else 'badge-bad'}'>{de_val:.0f}%</span>", unsafe_allow_html=True)

            peg = info.get('pegRatio', 0)
            peg_val = peg if peg is not None else 0
            h3.markdown(f"**PEG**<br><span class='{'badge-good' if peg_val < 1 and peg_val > 0 else 'badge-neutral'}'>{peg_val:.2f}</span>", unsafe_allow_html=True)

            pm = info.get('profitMargins', 0)
            h4.metric("Margin", f"{pm*100:.1f}%" if pm else "N/A")

            st.divider()
            
            # Valuation & Chart
            c_chart, c_val = st.columns([3, 1])
            with c_val:
                eps = info.get('trailingEps', 0)
                bv = info.get('bookValue', 0)
                if eps and bv and eps > 0 and bv > 0:
                    graham = math.sqrt(22.5 * eps * bv)
                    diff = ((curr - graham)/graham)*100
                    st.metric("Graham Value", f"{graham:.2f}")
                    if diff > 0: st.error(f"Overvalued {diff:.0f}%")
                    else: st.success(f"Undervalued {abs(diff):.0f}%")
            
            with c_chart:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='50 MA', line=dict(color='orange', width=1)))
                fig.update_layout(height=400, template="plotly_dark", title=f"{ticker} Trend")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} stock news")]))
        else:
            st.error("Data not found.")