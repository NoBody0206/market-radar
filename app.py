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
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(page_title="Executive Market Radar 13.0", layout="wide", page_icon="🦅")
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
    .verdict-pass { color: #4CAF50; font-weight: 900; font-size: 18px; border: 1px solid #4CAF50; padding: 5px 10px; border-radius: 5px; }
    .verdict-fail { color: #FF5252; font-weight: 900; font-size: 18px; border: 1px solid #FF5252; padding: 5px 10px; border-radius: 5px; }
    .verdict-neutral { color: #FFC107; font-weight: 900; font-size: 18px; border: 1px solid #FFC107; padding: 5px 10px; border-radius: 5px; }
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
        return stock.info, stock.history(period="1y"), stock.financials, stock.balance_sheet
    except: return None, None, None, None

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
st.title("🦅 Executive Market Radar 13.0")
st.caption("Strategy Scorecards | Volume Analysis | Automated Verdicts")

tab_india, tab_global, tab_ceo, tab_trade, tab_deep = st.tabs(["🇮🇳 India", "🌎 Global", "🏛️ CEO Radar", "📈 Trading Floor", "🧠 Analyst Lab"])

# --- TAB 1 & 2 & 3 (Standard) ---
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

# --- TAB 5: ANALYST LAB (UPGRADED) ---
with tab_deep:
    st.markdown("<div class='section-header'>🔍 Analyst Masterclass</div>", unsafe_allow_html=True)
    col_input, col_method = st.columns([2, 2])
    with col_input: ticker = st.text_input("Analyze Ticker:", "RELIANCE.NS").upper()
    with col_method: method = st.selectbox("Select Strategy:", ["🚀 CAN SLIM (Growth)", "🪄 Magic Formula (Value)", "🏰 MOAT Analysis", "🏦 CAMEL(S) (Banks)", "🏇 Jockey (Mgmt)", "🕵️ Scuttlebutt"])

    if ticker:
        info, hist, fin, bal = get_company_info(ticker)
        if info and not hist.empty:
            curr_price = hist['Close'].iloc[-1]
            st.metric(f"{info.get('shortName', ticker)}", f"{info.get('currency', '')} {curr_price:,.2f}")
            st.divider()

            # --- SCORECARD ENGINE ---
            score = 0
            max_score = 0
            verdict = "NEUTRAL"

            # HELPER FOR SAFE GET
            def s_get(d, k, fallback=0): return d.get(k, fallback) if d.get(k) is not None else fallback

            if "CAN SLIM" in method:
                max_score = 3
                eps_g = s_get(info, 'earningsGrowth')
                rev_g = s_get(info, 'revenueGrowth')
                high_52 = s_get(info, 'fiftyTwoWeekHigh')
                dist = (curr_price / high_52) * 100 if high_52 else 0
                
                if eps_g > 0.15: score += 1
                if rev_g > 0.15: score += 1
                if dist > 85: score += 1
                
                st.markdown("<div class='method-card'><h3>🚀 CAN SLIM Analysis</h3><p>Focus: High Growth, Momentum, Volume.</p></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("EPS Growth", f"{eps_g*100:.1f}%", delta="Target: >15%")
                c2.metric("Rev Growth", f"{rev_g*100:.1f}%", delta="Target: >15%")
                c3.metric("Near 52W High", f"{dist:.0f}%", delta="Target: >85%")

            elif "Magic Formula" in method:
                max_score = 2
                pe = s_get(info, 'trailingPE')
                ey = (1/pe * 100) if pe > 0 else 0
                roc = s_get(info, 'returnOnEquity')
                
                if ey > 4: score += 1
                if roc > 0.15: score += 1
                
                st.markdown("<div class='method-card'><h3>🪄 Magic Formula</h3><p>Focus: High Quality (ROC) at Low Price (Yield).</p></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("Earnings Yield", f"{ey:.2f}%", delta="Target: >4%")
                c2.metric("Return on Equity", f"{roc*100:.1f}%", delta="Target: >15%")

            elif "MOAT" in method:
                max_score = 3
                pm = s_get(info, 'grossMargins')
                roe = s_get(info, 'returnOnEquity')
                de = s_get(info, 'debtToEquity')
                
                if pm > 0.30: score += 1
                if roe > 0.15: score += 1
                if de < 50: score += 1
                
                st.markdown("<div class='method-card'><h3>🏰 MOAT Analysis</h3><p>Focus: Competitive Advantage & Safety.</p></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Gross Margin", f"{pm*100:.1f}%", delta="Target: >30%")
                c2.metric("ROE", f"{roe*100:.1f}%", delta="Target: >15%")
                c3.metric("Debt/Equity", f"{de:.0f}%", delta="Target: <50%", delta_color="inverse")

            # --- RENDER VERDICT ---
            if max_score > 0:
                v_col, msg_col = st.columns([1, 4])
                if score == max_score: 
                    verdict_class = "verdict-pass"
                    verdict_text = "PASS"
                elif score > 0:
                    verdict_class = "verdict-neutral"
                    verdict_text = "NEUTRAL"
                else:
                    verdict_class = "verdict-fail"
                    verdict_text = "FAIL"
                
                with v_col: st.markdown(f"<div class='{verdict_class}'>{verdict_text} ({score}/{max_score})</div>", unsafe_allow_html=True)
                with msg_col: st.caption("Automated Score based on framework criteria.")
                st.divider()

            # --- CHART WITH VOLUME (SUBPLOTS) ---
            st.subheader("Price & Volume Action")
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='50 MA', line=dict(color='orange')), row=1, col=1)
            
            # Volume Bar
            colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for index, row in hist.iterrows()]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color=colors), row=2, col=1)
            
            fig.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # News
            st.divider()
            render_news(fetch_feed_parallel([get_google_rss(f"{info.get('shortName', ticker)} stock news")]))