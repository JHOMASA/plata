import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time
import random
from sklearn.metrics import mean_absolute_error




# Configuration
FINGPT_API_KEY = "AIzaTRDjNFU6WAx6FJ74zhm2vQqWyD5MsYKUcOk"  # Replace with actual key
NEWS_API_KEY = "3f8e6bb1fb72490b835c800afcadd1aa"      # Replace with actual key


@lru_cache(maxsize=32)
def fetch_stock_data_cached(symbol: str, period: str = "1y"):
    """Fetch stock data with caching. Returns a tuple of (success, data_or_error)"""
    try:
        # Add delay to prevent rate limiting
        time.sleep(random.uniform(1, 2))
        
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period=period)
        if stock_data.empty:
            return False, "No data found for this ticker"
        
        # Convert DataFrame to JSON-serializable format for caching
        return True, stock_data.to_json(date_format='iso')
    except Exception as e:
        return False, f"Error fetching stock data: {e}"
def get_stock_data(symbol: str, period: str = "1y"):
    """Wrapper that handles the cached data conversion"""
    success, result = fetch_stock_data_cached(symbol, period)
    
    if success:
        # Convert JSON back to DataFrame
        return pd.read_json(result), None
    else:
        return pd.DataFrame(), result

# Enhanced visualization for all sections
def display_stock_analysis(stock_data, ticker):
    col1, col2 = st.columns(2)
    
    with col1:
        # Price History
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
        fig1.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Volume Analysis
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'))
        fig2.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Technical Indicators
    st.subheader("Technical Indicators")
    indicators = st.multiselect("Select indicators", 
                               ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
                               default=["SMA", "RSI"])
    
    if indicators:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
        
        if "SMA" in indicators:
            sma = stock_data['Close'].rolling(20).mean()
            fig3.add_trace(go.Scatter(x=stock_data.index, y=sma, name='20-day SMA'))
        
        if "RSI" in indicators:
            delta = stock_data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=rsi, name='RSI'))
            fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis_range=[0,100])
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        fig3.update_layout(title="Technical Indicators")
        st.plotly_chart(fig3, use_container_width=True)

def display_monte_carlo(simulations):
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulation Paths
        fig1 = go.Figure()
        for i in range(min(20, simulations.shape[1])):
            fig1.add_trace(go.Scatter(
                x=np.arange(simulations.shape[0]),
                y=simulations[:, i],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ))
        fig1.update_layout(title="Monte Carlo Simulation Paths", 
                          xaxis_title="Days", 
                          yaxis_title="Price")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Terminal Distribution
        terminal_prices = simulations[-1, :]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=terminal_prices, name="Outcomes"))
        fig2.update_layout(title="Terminal Price Distribution",
                          xaxis_title="Price",
                          yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Risk Metrics
    st.subheader("Risk Analysis")
    var_95 = np.percentile(terminal_prices, 5)
    var_99 = np.percentile(terminal_prices, 1)
    
    metrics = pd.DataFrame({
        "Metric": ["5% VaR", "1% VaR", "Expected Value", "Best Case", "Worst Case"],
        "Value": [
            f"${var_95:.2f}", 
            f"${var_99:.2f}",
            f"${terminal_prices.mean():.2f}",
            f"${terminal_prices.max():.2f}",
            f"${terminal_prices.min():.2f}"
        ]
    })
    st.table(metrics)

def display_financial_ratios(ratios, ticker):
    # Benchmark data (mock)
    sector_avg = {
        'P/E Ratio': 15.2,
        'Volatility': 12.5,
        'Sharpe Ratio': 1.3,
        'Max Drawdown': 8.2
    }
    
    # Create comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(ratios.keys()),
        y=[float(r.strip('%')) if '%' in r else float(r) for r in ratios.values()],
        name=ticker,
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=list(sector_avg.keys()),
        y=list(sector_avg.values()),
        name='Sector Average',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Financial Ratio Comparison",
        barmode='group',
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Ratio Interpretation
    st.subheader("Ratio Analysis")
    
    if float(ratios.get('P/E Ratio', 0)) > sector_avg['P/E Ratio']:
        st.warning(f"P/E Ratio ({ratios['P/E Ratio']}) is higher than sector average ({sector_avg['P/E Ratio']})")
    else:
        st.success(f"P/E Ratio ({ratios['P/E Ratio']}) is favorable compared to sector average ({sector_avg['P/E Ratio']})")
    
    if float(ratios.get('Volatility', '0%').strip('%')) > sector_avg['Volatility']:
        st.warning(f"Higher volatility ({ratios['Volatility']}) than sector average ({sector_avg['Volatility']}%)")
    else:
        st.success(f"Lower volatility ({ratios['Volatility']}) than sector average ({sector_avg['Volatility']}%)")

def display_predictions(historical_data, predictions, model_name):
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    # Predictions
    future_dates = pd.date_range(
        start=historical_data.index[-1],
        periods=len(predictions)+1
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name=f'{model_name} Forecast',
        line=dict(color='green', dash='dot')
    ))
    
    # Confidence Interval (if available)
    if hasattr(predictions, 'conf_int'):
        ci = predictions.conf_int()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ci.iloc[:, 0],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ci.iloc[:, 1],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title=f"{model_name} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Metrics
    if len(historical_data) > 30:  # Only show if sufficient history
        test = historical_data['Close'].values[-30:]
        mae = mean_absolute_error(test, predictions[:30])
        st.metric("Mean Absolute Error (30-day backtest)", f"${mae:.2f}")

# Updated main app structure
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Advanced Stock Analysis Dashboard")
    
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Stock Analysis", "Monte Carlo", "Financial Ratios", "Predictions"]
    )
    
    # Ticker Input
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").strip().upper()

    if not ticker:
        st.error("Please enter a valid ticker symbol")
        return
        
    # Date Range Selector
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Fetch Data
    data,error = get_stock_data(ticker, period)
    
    if data.empty:
       if "Rate limited" in error:
          st.error("""
          ⚠️ Yahoo Finance rate limit reached. 
          Please wait a few minutes and try again, or:
          - Use the cached version if available
          - Try again later
          """)
          st.stop()  # Prevents the app from trying to continue
       else:
           st.error(f"Error fetching data: {error}")
       return
    
    # Analysis Sections
    if analysis_type == "Stock Analysis":
        display_stock_analysis(data, ticker)
        
    elif analysis_type == "Monte Carlo":
        st.header("🎲 Monte Carlo Simulation")
        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
        time_horizon = st.slider("Time Horizon (days)", 30, 365, 180)
        
        if st.button("Run Simulation"):
            simulations = monte_carlo_simulation(data, n_simulations, time_horizon)
            display_monte_carlo(simulations)
    
    elif analysis_type == "Financial Ratios":
        st.header("📈 Financial Ratios Analysis")
        ratios = calculate_risk_metrics(data)
        display_financial_ratios(ratios, ticker)
    
    elif analysis_type == "Predictions":
        st.header("🔮 Price Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Prediction Model",
                ["Holt-Winters", "Prophet", "LSTM", "ARIMA", "XGBoost"]
            )
            
        with col2:
            if model_type == "Holt-Winters":
                seasonality = st.radio(
                    "Seasonality",
                    ["Weekly (5)", "Monthly (21)", "Quarterly (63)"],
                    horizontal=True
                )
                seasonal_periods = int(seasonality.split("(")[1].replace(")", ""))
        
        if st.button("Generate Predictions"):
            with st.spinner(f"Training {model_type} model..."):
                if model_type == "Holt-Winters":
                    model, error = train_holt_winters(data, seasonal_periods)
                    if model is None:
                        st.error(error)
                    else:
                        predictions = predict_holt_winters(30)
                        display_predictions(data, predictions, "Holt-Winters")
                elif model_type == "Prophet":
                    model = train_prophet_model(data)
                    predictions = predict_prophet(model)
                elif model_type == "LSTM":
                    model, scaler = train_lstm_model(data)
                    predictions = predict_lstm(model, scaler, data)
                
                display_predictions(data, predictions, model_type)

if __name__ == "__main__":
    main()
