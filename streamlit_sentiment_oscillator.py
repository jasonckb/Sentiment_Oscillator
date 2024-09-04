import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial

# 配置
st.set_page_config(layout="wide")

# 常量
HK_SYMBOLS = ['0001.HK', '0005.HK', '0011.HK', '0388.HK', '0700.HK', '0941.HK', '1299.HK', '2318.HK', '3988.HK']
US_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

# 數據處理函數
@st.cache_data
def load_data(symbols):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(partial(get_stock_data_with_sentiment, period="1y"), symbols))
    return pd.DataFrame(results).set_index('symbol')

def get_stock_data_with_sentiment(symbol, period="1y"):
    data = get_stock_data(symbol, period)
    if data.empty:
        return {'symbol': symbol, 'sentiment': np.nan, 'prev_sentiment': np.nan, 'last_close': np.nan, 'last_date': None}
    
    sentiment = calculate_sentiment_oscillator(data)
    return {
        'symbol': symbol,
        'sentiment': sentiment.iloc[-1],
        'prev_sentiment': sentiment.iloc[-2] if len(sentiment) > 1 else np.nan,
        'last_close': data['Close'].iloc[-1],
        'last_date': data.index[-1]
    }

def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data.dropna()

def calculate_sentiment_oscillator(data):
    rsi = calculate_rsi(data)
    stoch = calculate_stochastic(data)
    cci = calculate_cci(data)
    bbp = calculate_bbp(data)
    ma = calculate_ma(data)
    supertrend = calculate_supertrend(data)
    lr = calculate_linear_regression(data)
    ms = calculate_market_structure(data)
    
    sentiment = (rsi + stoch + cci + bbp + ma + supertrend + lr + ms) / 8
    return sentiment.rolling(window=3).mean()

# 在這裡添加您的其他函數，如 calculate_rsi, calculate_stochastic 等

# UI 組件
def render_stock_grid(sorted_sentiment):
    st.markdown("""
    <style>
    .stock-button {
        width: 100px;
        height: 60px;
        padding: 5px 2px;
        margin: 2px;
        white-space: normal;
        word-wrap: break-word;
        font-size: 12px;
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        border: none;
    }
    .stock-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    grid_html = '<div class="stock-grid">'
    for symbol, data in sorted_sentiment.iterrows():
        sentiment_value = data['sentiment']
        button_color = get_button_color(sentiment_value)
        text_color = get_text_color(sentiment_value)
        display_value = f'{sentiment_value:.2f}' if pd.notna(sentiment_value) and np.isfinite(sentiment_value) else 'N/A'
        
        grid_html += f"""
        <button class="stock-button" style="background-color: {button_color}; color: {text_color};"
                onclick="Streamlit.setComponentValue('{symbol}');">
            {symbol}<br>{display_value}
        </button>
        """
    grid_html += '</div>'

    st.markdown(grid_html, unsafe_allow_html=True)

def render_stock_chart(symbol, sentiment_data):
    st.subheader(f"Detailed Chart for {symbol}")
    try:
        with st.spinner(f"Loading chart for {symbol}..."):
            chart = plot_chart(symbol)
            st.plotly_chart(chart, use_container_width=True)
            
            symbol_data = sentiment_data.loc[symbol]
            st.write(f"Last Close: {symbol_data['last_close']:.2f}")
            st.write(f"Last Date: {symbol_data['last_date']}")
            st.write(f"Current Sentiment: {symbol_data['sentiment']:.2f}")
    except Exception as e:
        st.error(f"Error generating chart for {symbol}: {str(e)}")

# 主函數
def main():
    st.title("Stock Sentiment Oscillator Dashboard")

    # 側邊欄
    st.sidebar.header("Stock Universe Selection")
    market = st.sidebar.radio("Select Market", ["HK Stock", "US Stock"])
    symbols = HK_SYMBOLS if market == "HK Stock" else US_SYMBOLS

    # 主要內容區域
    main_container = st.container()
    
    with main_container:
        # 加載數據
        with st.spinner("Loading data..."):
            sentiment_data = load_data(symbols)
        
        sorted_sentiment = sentiment_data.sort_values('sentiment', ascending=False)

        # 顯示信號和超買超賣股票
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stocks with Buy Signal:")
            buy_signals = sorted_sentiment[(sorted_sentiment['prev_sentiment'] < 50) & (sorted_sentiment['sentiment'] >= 50)]
            st.write(", ".join(buy_signals.index) if not buy_signals.empty else "Nil")
            
            st.subheader("Stocks Overbought:")
            overbought_stocks = sorted_sentiment[sorted_sentiment['sentiment'] > 75]
            st.write(", ".join(overbought_stocks.index) if not overbought_stocks.empty else "Nil")

        with col2:
            st.subheader("Stocks with Sell Signal:")
            sell_signals = sorted_sentiment[(sorted_sentiment['prev_sentiment'] >= 50) & (sorted_sentiment['sentiment'] < 50)]
            st.write(", ".join(sell_signals.index) if not sell_signals.empty else "Nil")
            
            st.subheader("Stocks Oversold:")
            oversold_stocks = sorted_sentiment[sorted_sentiment['sentiment'] < 25]
            st.write(", ".join(oversold_stocks.index) if not oversold_stocks.empty else "Nil")

        # 股票按鈕網格
        render_stock_grid(sorted_sentiment)

        # 處理按鈕點擊
        clicked_symbol = st.experimental_get_query_params().get('clicked_stock', [None])[0]
        if clicked_symbol:
            render_stock_chart(clicked_symbol, sentiment_data)

        # 刷新按鈕
        if 'refresh_key' not in st.session_state:
            st.session_state.refresh_key = 0
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.session_state.refresh_key += 1
        
        st.text(f"Refresh key: {st.session_state.refresh_key}")

    st.markdown("---")
    st.markdown("Data provided by Yahoo Finance. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()