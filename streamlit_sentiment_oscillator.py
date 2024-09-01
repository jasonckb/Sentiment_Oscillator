import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data.dropna()

def interpolate(value, value_high, value_low, range_high, range_low):
    return range_low + (value - value_low) * (range_high - range_low) / (value_high - value_low)

def normalize(series, buy, sell, smooth):
    os = pd.Series(0, index=series.index)
    os[buy] = 1
    os[sell] = -1
    
    max_val = series.copy()
    min_val = series.copy()
    
    for i in range(1, len(series)):
        if os.iloc[i] > os.iloc[i-1]:
            max_val.iloc[i] = series.iloc[i]
            min_val.iloc[i] = min_val.iloc[i-1]
        elif os.iloc[i] < os.iloc[i-1]:
            min_val.iloc[i] = series.iloc[i]
            max_val.iloc[i] = max_val.iloc[i-1]
        else:
            max_val.iloc[i] = max(series.iloc[i], max_val.iloc[i-1])
            min_val.iloc[i] = min(series.iloc[i], min_val.iloc[i-1])
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized.rolling(window=smooth).mean() * 100

def calculate_rsi(data, length=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    result = pd.Series(index=rsi.index)
    result[rsi > 70] = interpolate(rsi[rsi > 70], 100, 70, 100, 75)
    result[(rsi > 50) & (rsi <= 70)] = interpolate(rsi[(rsi > 50) & (rsi <= 70)], 70, 50, 75, 50)
    result[(rsi > 30) & (rsi <= 50)] = interpolate(rsi[(rsi > 30) & (rsi <= 50)], 50, 30, 50, 25)
    result[rsi <= 30] = interpolate(rsi[rsi <= 30], 30, 0, 25, 0)
    
    return result

def calculate_stochastic(data, k_window=14, smooth_k=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=smooth_k).mean()
    
    result = pd.Series(index=d.index)
    result[d > 80] = interpolate(d[d > 80], 100, 80, 100, 75)
    result[(d > 50) & (d <= 80)] = interpolate(d[(d > 50) & (d <= 80)], 80, 50, 75, 50)
    result[(d > 20) & (d <= 50)] = interpolate(d[(d > 20) & (d <= 50)], 50, 20, 50, 25)
    result[d <= 20] = interpolate(d[d <= 20], 20, 0, 25, 0)
    
    return result

def calculate_cci(data, length=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=length).mean()
    mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    
    result = pd.Series(index=cci.index)
    result[cci > 100] = interpolate(cci[cci > 100], 300, 100, 100, 75)
    result[(cci >= 0) & (cci <= 100)] = interpolate(cci[(cci >= 0) & (cci <= 100)], 100, 0, 75, 50)
    result[(cci < 0) & (cci >= -100)] = interpolate(cci[(cci < 0) & (cci >= -100)], 0, -100, 50, 25)
    result[cci < -100] = interpolate(cci[cci < -100], -100, -300, 25, 0)
    
    return result

def calculate_bbp(data, length=13):
    ma = data['Close'].ewm(span=length, adjust=False).mean()
    bbp = data['High'] + data['Low'] - 2 * ma
    bbp_std = bbp.rolling(window=100).std()
    upper = bbp.rolling(window=100).mean() + 2 * bbp_std
    lower = bbp.rolling(window=100).mean() - 2 * bbp_std
    
    result = pd.Series(index=bbp.index)
    result[bbp > upper] = interpolate(bbp[bbp > upper], 1.5 * upper, upper, 100, 75)
    result[(bbp > 0) & (bbp <= upper)] = interpolate(bbp[(bbp > 0) & (bbp <= upper)], upper, 0, 75, 50)
    result[(bbp < 0) & (bbp >= lower)] = interpolate(bbp[(bbp < 0) & (bbp >= lower)], 0, lower, 50, 25)
    result[bbp < lower] = interpolate(bbp[bbp < lower], lower, 1.5 * lower, 25, 0)
    
    return result

def calculate_ma(data, length=20, ma_type='SMA'):
    if ma_type == 'SMA':
        ma = data['Close'].rolling(window=length).mean()
    elif ma_type == 'EMA':
        ma = data['Close'].ewm(span=length, adjust=False).mean()
    
    return normalize(data['Close'], data['Close'] > ma, data['Close'] < ma, 3)

def calculate_supertrend(data, factor=3, period=10):
    hl2 = (data['High'] + data['Low']) / 2
    atr = data['High'].sub(data['Low']).rolling(window=period).mean()
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)
    
    for i in range(period, len(data)):
        if data['Close'].iloc[i] > upperband.iloc[i-1]:
            supertrend.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        elif data['Close'].iloc[i] < lowerband.iloc[i-1]:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1 and lowerband.iloc[i] < supertrend.iloc[i]:
                supertrend.iloc[i] = lowerband.iloc[i]
            if direction.iloc[i] == -1 and upperband.iloc[i] > supertrend.iloc[i]:
                supertrend.iloc[i] = upperband.iloc[i]
    
    return normalize(data['Close'], data['Close'] > supertrend, data['Close'] < supertrend, 3)

def calculate_linear_regression(data, length=25):
    lr = 50 * data['Close'].rolling(window=length).apply(lambda x: np.corrcoef(x, range(length))[0, 1]) + 50
    return lr

def calculate_market_structure(data, length=5):
    highs = data['High'].rolling(window=length).max()
    lows = data['Low'].rolling(window=length).min()
    
    bull_break = (data['Close'] > highs.shift(1)) & (data['Close'].shift(1) <= highs.shift(1))
    bear_break = (data['Close'] < lows.shift(1)) & (data['Close'].shift(1) >= lows.shift(1))
    
    return normalize(data['Close'], bull_break, bear_break, 3)

def calculate_volume_profile(data, bins=40):
    price_range = data['Close'].max() - data['Close'].min()
    bin_size = price_range / bins
    price_bins = pd.cut(data['Close'], bins=bins)
    volume_profile = data.groupby(price_bins)['Volume'].sum()
    bin_centers = [(i.left + i.right) / 2 for i in volume_profile.index]
    
    poc_price = bin_centers[volume_profile.argmax()]
    
    total_volume = volume_profile.sum()
    target_volume = total_volume * 0.7
    cumulative_volume = 0
    value_area_low = value_area_high = poc_price
    
    for price, volume in zip(bin_centers, volume_profile):
        cumulative_volume += volume
        if cumulative_volume <= target_volume / 2:
            value_area_low = price
        if cumulative_volume >= total_volume - target_volume / 2:
            value_area_high = price
            break
    
    return volume_profile, bin_centers, bin_size, poc_price, value_area_low, value_area_high

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

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
    return sentiment

def get_button_color(value):
    if pd.isna(value) or not np.isfinite(value):
        return "rgb(128, 128, 128)"
    value = max(0, min(100, value))
    if value > 50:
        green = int(255 * (value - 50) / 50)
        return f"rgb(0, {green}, 0)"
    else:
        red = int(255 * (50 - value) / 50)
        return f"rgb({red}, 0, 0)"

def get_text_color(value):
    if pd.isna(value) or not np.isfinite(value):
        return "white"
    value = max(0, min(100, value))
    if value > 75:
        return "red"
    elif value < 25:
        return "blue"
    else:
        return "black"

def plot_chart(ticker):
    data = get_stock_data(ticker, period="2y")
    one_year_ago = data.index[-1] - pd.DateOffset(years=1)
    data_to_plot = data.loc[one_year_ago:]
    sentiment = calculate_sentiment_oscillator(data)
    sentiment_to_plot = sentiment.loc[one_year_ago:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=data_to_plot.index,
        open=data_to_plot['Open'],
        high=data_to_plot['High'],
        low=data_to_plot['Low'],
        close=data_to_plot['Close'],
        name='Price',
        increasing_line_color='dodgerblue',
        decreasing_line_color='red'
    ), row=1, col=1)
    
    # Calculate EMAs
    ema_20 = calculate_ema(data, 20)
    ema_50 = calculate_ema(data, 50)
    ema_200 = calculate_ema(data, 200)

    # Calculate the position for price annotations
    first_date = data.index[0]
    last_date = data.index[-1]
    annotation_x = last_date + pd.Timedelta(days=2)  # 2 days after the last candle
    mid_date = first_date + (last_date - first_date) / 2  # Middle of the date range         
    
    # Add EMA lines
    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=ema_20.iloc[-1], y1=ema_20.iloc[-1],
                  line=dict(color="gray", width=1, dash="dash"))
    fig.add_annotation(x=annotation_x, y=ema_20.iloc[-1], text=f"20 EMA: {ema_20.iloc[-1]:.2f}",
                       showarrow=False, xanchor="left", font=dict(size=12, color="gray"))

    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=ema_50.iloc[-1], y1=ema_50.iloc[-1],
                  line=dict(color="gray", width=2, dash="dash"))
    fig.add_annotation(x=annotation_x, y=ema_50.iloc[-1], text=f"50 EMA: {ema_50.iloc[-1]:.2f}",
                       showarrow=False, xanchor="left", font=dict(size=12, color="gray"))

    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=ema_200.iloc[-1], y1=ema_200.iloc[-1],
                  line=dict(color="gray", width=3, dash="dash"))
    fig.add_annotation(x=annotation_x, y=ema_200.iloc[-1], text=f"200 EMA: {ema_200.iloc[-1]:.2f}",
                       showarrow=False, xanchor="left", font=dict(size=12, color="gray"))

    # Add current price annotation
    current_price = data['Close'].iloc[-1]
    fig.add_annotation(x=annotation_x, y=current_price, text=f"Current Price: {current_price:.2f}",
                       showarrow=False, xanchor="left", font=dict(size=14, color="black"))

    # Calculate and add volume profile
    volume_profile, bin_centers, bin_size, poc_price, value_area_low, value_area_high = calculate_volume_profile(data)
    max_volume = volume_profile.max()
    fig.add_trace(go.Bar(
        x=volume_profile.values,
        y=bin_centers,
        orientation='h',
        name='Volume Profile',
        marker_color='rgba(200, 200, 200, 0.5)',
        width=bin_size,
        xaxis='x2'
    ))

    # Add POC line (red)
    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=poc_price, y1=poc_price,
                  line=dict(color="red", width=4))
    fig.add_annotation(x=annotation_x, y=poc_price, text=f"POC: {poc_price:.2f}",
                       showarrow=False, xanchor="left", font=dict(size=12, color="red"))

    # Add Value Area lines (purple) with labels above and below the lines
    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=value_area_low, y1=value_area_low,
                  line=dict(color="purple", width=2))
    fig.add_annotation(x=mid_date, y=value_area_low, text=f"Value at Low: {value_area_low:.2f}",
                       showarrow=False, xanchor="center", yanchor="top", font=dict(size=12, color="purple"),
                       yshift=-5)  # Shift the label 5 pixels below the line

    fig.add_shape(type="line", x0=first_date, x1=annotation_x, y0=value_area_high, y1=value_area_high,
                  line=dict(color="purple", width=2))
    fig.add_annotation(x=mid_date, y=value_area_high, text=f"Value at High: {value_area_high:.2f}",
                       showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=12, color="purple"),
                       yshift=5)  # Shift the label 5 pixels above the line
    
    fig.add_trace(go.Scatter(
        x=sentiment_to_plot.index,
        y=sentiment_to_plot, 
        line=dict(color='purple', width=2),
        name='Sentiment Oscillator'
    ), row=2, col=1)
    
    fig.add_traces([
        go.Scatter(
            x=sentiment_to_plot.index,
            y=sentiment_to_plot.where(sentiment_to_plot > 50, 50), 
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.2)', 
            line=dict(color='rgba(0,0,0,0)'),
            name='Bullish'
        ),
        go.Scatter(
            x=sentiment_to_plot.index,
            y=sentiment_to_plot.where(sentiment_to_plot < 50, 50), 
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)', 
            line=dict(color='rgba(0,0,0,0)'),
            name='Bearish'
        )
    ], rows=[2,2], cols=[1,1])
    
    fig.add_hline(y=75, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        title=f"{ticker} - Price Chart and Sentiment Oscillator",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=800,
        width=1200,
        showlegend=False,
        xaxis=dict(
            showline=True,
            showgrid=True,
            tickformat="%Y-%m-%d",
            type="date"
        ),
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Oscillator", range=[0, 100], row=2, col=1)
    
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=["2023-12-25", "2024-01-01"])
        ],
        range=[first_date, annotation_x]
    )
    
    return fig

# Streamlit app
st.set_page_config(layout="wide")
st.title("Stock Sentiment Oscillator Dashboard")

# Define the US and HK symbols
us_symbols = [
    '^NDX', '^GSPC', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'TSLA', 'JPM',
    'V', 'UNH', 'LLY', 'JNJ', 'XOM', 'WMT', 'MA', 'PG', 'KO', 'HD',
    'AVGO', 'CVX', 'MRK', 'PE', 'GS', 'ABBV', 'COST', 'TSM', 'VZ', 'PFE',
    'NFLX', 'ADBE', 'ASML', 'CRM', 'ACN', 'TRV', 'BA', 'TXN', 'IBM', 'DIS',
    'UPS', 'SPGI', 'INTC', 'AMD', 'QCOM', 'AMT', 'CHTR', 'SBUX', 'MS', 'BLK',
    'GE', 'MMM', 'GILD', 'CAT', 'INTU', 'ISRG', 'AMGN', 'CVS', 'DE', 'EQIX',
    'TJX', 'PGR', 'BKNG', 'MU', 'LRCX', 'REGN', 'ARM', 'PLTR', 'SNOW', 'PANW',
    'CRWD', 'ZS', 'ABNB', 'CDNS', 'DDOG', 'ICE', 'TTD', 'TEAM', 'CEG', 'VST',
    'NRG', 'NEE', 'PYPL', 'FTNT', 'IDXX', 'SMH', 'XLU', 'XLP', 'XLE', 'XLK',
    'XLY', 'XLI', 'XLB', 'XLRE', 'XLF', 'XLV', 'OXY', 'NVO', 'CCL', 'LEN'
]

hk_symbols = [
    '^HSI', '0020.HK', '0017.HK', '0241.HK', '0066.HK', '1038.HK', '0006.HK', '0011.HK', '0012.HK', '0857.HK',
    '3988.HK', '1044.HK', '0386.HK', '2388.HK', '1113.HK', '0941.HK', '1997.HK', '0001.HK', '1093.HK', '1109.HK',
    '1177.HK', '1211.HK', '1299.HK', '1398.HK', '0016.HK', '0175.HK', '1810.HK', '1876.HK', '1928.HK', '2007.HK',
    '2018.HK', '2269.HK', '2313.HK', '2318.HK', '2319.HK', '2331.HK', '2382.HK', '2628.HK', '0267.HK', '0027.HK',
    '0288.HK', '0003.HK', '3690.HK', '0388.HK', '3968.HK', '0005.HK', '6098.HK', '0669.HK', '6862.HK', '0688.HK',
    '0700.HK', '0762.HK', '0823.HK', '0868.HK', '0883.HK', '0939.HK', '0960.HK', '0968.HK', '9988.HK', '1024.HK',
    '1347.HK', '1833.HK', '2013.HK', '2518.HK', '0268.HK', '0285.HK', '3888.HK', '0522.HK', '6060.HK', '6618.HK',
    '6690.HK', '0772.HK', '0909.HK', '9618.HK', '9626.HK', '9698.HK', '0981.HK', '9888.HK', '0992.HK', '9961.HK',
    '9999.HK', '2015.HK', '0291.HK', '0293.HK', '0358.HK', '1772.HK', '1776.HK', '1787.HK', '1801.HK', '1818.HK',
    '1898.HK', '0019.HK', '1929.HK', '0799.HK', '0836.HK', '0853.HK', '0914.HK', '0916.HK', '6078.HK', '2333.HK', '3888.HK'
]

# Sidebar
st.sidebar.header("Stock Universe Selection")
market = st.sidebar.radio("Select Market", ["HK Stock", "US Stock"])

if market == "HK Stock":
    symbols = hk_symbols
else:
    symbols = us_symbols

# Main app
@st.cache_data
def load_data(symbols):
    data = {}
    for symbol in symbols:
        try:
            stock_data = get_stock_data(symbol, period="2y")
            sentiment = calculate_sentiment_oscillator(stock_data)
            data[symbol] = sentiment.iloc[-1]
        except Exception as e:
            st.warning(f"Error loading data for {symbol}: {e}")
    return pd.Series(data)

with st.spinner("Loading data..."):
    sentiment_data = load_data(symbols)

# Sort the sentiment data
sorted_sentiment = sentiment_data.sort_values(ascending=False)

# Calculate buy and sell signals
yesterday_sentiment = sentiment_data.shift(1)
buy_signals = sorted_sentiment[(yesterday_sentiment < 50) & (sorted_sentiment >= 50)]
sell_signals = sorted_sentiment[(yesterday_sentiment >= 50) & (sorted_sentiment < 50)]

# Display buy signals
st.subheader("Stocks with Buy Signal:")
if not buy_signals.empty:
    st.write(", ".join(buy_signals.index))
else:
    st.write("Nil")

# Display sell signals
st.subheader("Stocks with Sell Signal:")
if not sell_signals.empty:
    st.write(", ".join(sell_signals.index))
else:
    st.write("Nil")

# Initialize session state to store the clicked symbol
if 'clicked_symbol' not in st.session_state:
    st.session_state.clicked_symbol = None

# Create a grid of 15 columns
# Create a grid of 15 columns
cols = st.columns(15)

# Custom CSS for button styling
st.markdown("""
<style>
div.stButton > button:first-child {
    height: auto;
    padding: 10px 5px;
    white-space: normal;
    word-wrap: break-word;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Display the sorted sentiment data in a grid
for i, (symbol, value) in enumerate(sorted_sentiment.items()):
    col = cols[i % 15]
    
    # Handle potential NaN or infinite values
    if pd.isna(value) or not np.isfinite(value):
        button_color = "rgb(128, 128, 128)"  # Gray for invalid values
        text_color = "white"
        display_value = 'N/A'
    else:
        button_color = get_button_color(value)
        text_color = get_text_color(value)
        display_value = f'{value:.2f}'
    
    # Create a unique key for each button
    button_key = f"btn_{symbol}"
    
    # Apply custom styling to the button
    custom_css = f"""
    <style>
    div[data-testid="stButton"] > button:first-child[key="{button_key}"] {{
        background-color: {button_color};
        color: {text_color};
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Create the button
    if col.button(f"{symbol}\n{display_value}", key=button_key):
        st.subheader(f"Detailed Chart for {symbol}")
        try:
            with st.spinner(f"Loading chart for {symbol}..."):
                chart = plot_chart(symbol)
                st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart for {symbol}: {str(e)}")

# Add a button to refresh the data
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Data provided by Yahoo Finance. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

