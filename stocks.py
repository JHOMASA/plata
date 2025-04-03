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
from functools import lru_cache
import requests
from typing import  Dict, Any,Tuple
from sklearn.preprocessing import MinMaxScaler


# Configuration

FMP_API_KEY = "7ZQp6gRCjJ1vucRutlFeRlpWjzh5GMhu"
FINGPT_API_KEY = "AIzaTRDjNFU6WAx6FJ74zhm2vQqWyD5MsYKUcOk"  # Replace with actual key
NEWS_API_KEY = "3f8e6bb1fb72490b835c800afcadd1aa"      # Replace with actual key


st.set_page_config(layout="wide")
st.title("📊 Advanced Stock Analysis Dashboard")
@lru_cache(maxsize=32)
def fetch_stock_data_yahoo(symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    """Fetch stock data from Yahoo Finance API"""
    try:
        # Map period to Yahoo Finance format
        period_map = {
            "1mo": "1mo",
            "3mo": "3mo",
            "6mo": "6mo",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y"
        }
        
        yahoo_period = period_map.get(period, "1y")
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yahoo_period)
        
        if df.empty:
            return pd.DataFrame(), "No data available for this symbol"
            
        # Clean and standardize the data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {str(e)}"

@lru_cache(maxsize=32)
def fetch_stock_data_cached(symbol: str, period: str = "1y") -> Tuple[bool, str]:
    """Fetch stock data with caching"""
    try:
        df, error = fetch_stock_data_yahoo(symbol, period)
        if error:
            return False, error
        return True, df.to_json(date_format='iso')
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_stock_data(symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    """Main function to get stock data"""
    success, result = fetch_stock_data_cached(symbol, period)
    if success:
        return pd.read_json(result), None
    return pd.DataFrame(), result

def get_yahoo_ratios(ticker: str) -> Dict[str, Any]:
    """Get financial ratios from Yahoo Finance"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        if not info:
            st.error("No financial data available for this ticker")
            return None
            
        # Extract relevant ratios
        ratios = {
            'priceEarningsRatio': info.get('trailingPE'),
            'priceToBookRatio': info.get('priceToBook'),
            'debtEquityRatio': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'returnOnEquity': info.get('returnOnEquity'),
            'returnOnAssets': info.get('returnOnAssets')
        }
        
        return ratios
        
    except Exception as e:
        st.error(f"Error fetching ratios: {str(e)}")
        return None


def calculate_risk_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate financial ratios from stock data that align with FMP API structure
    Args:
        data: DataFrame containing stock price history with columns ['Close', 'Volume', etc.]
    Returns:
        Dictionary of calculated ratios matching FMP's field names
    """
    try:
        if data.empty:
            return {}
        
        ratios = {}
        
        # Calculate volatility (annualized)
        returns = np.log(1 + data['Close'].pct_change())
        if not returns.empty:
            ratios['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate max drawdown
        rolling_max = data['Close'].cummax()
        daily_drawdown = data['Close']/rolling_max - 1
        ratios['maximumDrawdown'] = daily_drawdown.min()
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        if not returns.empty and returns.std() != 0:
            ratios['sharpeRatio'] = returns.mean() / returns.std() * np.sqrt(252)
        
        # Additional metrics that would come from FMP's fundamental data
        # These would normally come from FMP's API but we include placeholders
        ratios['priceEarningsRatio'] = None  # Would come from FMP
        ratios['priceToBookRatio'] = None    # Would come from FMP
        ratios['debtEquityRatio'] = None     # Would come from FMP
        ratios['currentRatio'] = None        # Would come from FMP
        ratios['returnOnEquity'] = None      # Would come from FMP
        ratios['returnOnAssets'] = None      # Would come from FMP
        
        # Convert all values to appropriate types
        for k, v in ratios.items():
            if v is not None:
                if k in ['volatility', 'maximumDrawdown', 'returnOnEquity', 'returnOnAssets']:
                    ratios[k] = float(v)
                else:
                    ratios[k] = float(v) if not pd.isna(v) else None
        
        return ratios
        
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")
        return {}

def monte_carlo_simulation(data: pd.DataFrame, n_simulations: int = 1000, days: int = 180) -> dict:
    """
    Enhanced Monte Carlo simulation with moving average smoothing options
    Returns dictionary containing:
    - raw_simulations: Original simulation paths
    - ma_simulations: Moving average smoothed paths
    - wma_simulations: Weighted moving average smoothed paths
    """
    try:
        # Calculate daily returns
        returns = np.log(1 + data['Close'].pct_change())
        mu = returns.mean()
        sigma = returns.std()
        last_price = data['Close'].iloc[-1]
        
        # Generate random walks
        raw_simulations = np.zeros((days, n_simulations))
        raw_simulations[0] = last_price
        
        for day in range(1, days):
            shock = np.random.normal(mu, sigma, n_simulations)
            raw_simulations[day] = raw_simulations[day-1] * np.exp(shock)
        
        # Apply smoothing techniques
        window_size = min(20, days//10)  # Adaptive window size
        
        # Simple Moving Average
        ma_simulations = np.zeros_like(raw_simulations)
        for i in range(n_simulations):
            ma_simulations[:, i] = pd.Series(raw_simulations[:, i]).rolling(window=window_size).mean().values
        
        # Weighted Moving Average (linear weights)
        wma_simulations = np.zeros_like(raw_simulations)
        weights = np.arange(1, window_size+1)  # Linear weights [1, 2, 3, ...]
        weights = weights / weights.sum()      # Normalized
        
        for i in range(n_simulations):
            series = pd.Series(raw_simulations[:, i])
            wma_simulations[:, i] = series.rolling(window=window_size)\
                                         .apply(lambda x: np.sum(weights * x))
        
        return {
            'raw': raw_simulations,
            'ma': ma_simulations,
            'wma': wma_simulations
        }
        
    except Exception as e:
        raise Exception(f"Enhanced Monte Carlo simulation failed: {str(e)}")
def train_holt_winters(data: pd.DataFrame, seasonal_periods: int) -> Tuple[object, str]:
    """Train Holt-Winters forecasting model"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(
            data['Close'],
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        ).fit()
        return model, None
    except Exception as e:
        return None, f"Holt-Winters training failed: {str(e)}"

def predict_holt_winters(model, periods: int = 30) -> pd.Series:
    """Generate predictions using Holt-Winters model"""
    try:
        return model.forecast(periods)
    except Exception as e:
        raise Exception(f"Holt-Winters prediction failed: {str(e)}")

def create_lagged_features(data: pd.DataFrame, lags: int = 30) -> pd.DataFrame:
    """Create lagged features for time series forecasting"""
    df = data.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

def train_random_forest(data: pd.DataFrame) -> object:
    """Train Random Forest model for time series forecasting"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Create lagged features
        df = create_lagged_features(data)
        
        # Prepare features and target
        X = df.drop(columns=['Close'])
        y = df['Close']
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        return model
        
    except Exception as e:
        raise Exception(f"Random Forest training failed: {str(e)}")

def predict_random_forest(model, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate predictions using Random Forest"""
    try:
        # Create recursive predictions
        current_data = data.copy()
        if 'Date' in current_data.columns:
            current_data = current_data.set_index('Date')
        predictions = []
        last_date = current_data.index[-1]
        for _ in range(periods):
            # Create features for next prediction
            latest = pd.DataFrame(index=[current_data.index[-1] + pd.Timedelta(days=1)])
            for i in range(1, 31):
                latest[f'lag_{i}'] = current_data['Close'].iloc[-i]
            
            # Make prediction
            pred = model.predict([latest_features])[0]
            predictions.append(pred)
            
            # Update data with new prediction
            new_row = pd.DataFrame({'Close':[pred]}, index = [last_date])
            current_data = pd.concat([current_data, new_row])
        
        return np.array(predictions)
        
    except Exception as e:
        raise Exception(f"Random Forest prediction failed: {str(e)}")
def train_lstm_model(data: pd.DataFrame) -> Tuple[object, object]:
    """Basic LSTM model training"""
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Close']].values)
        
        # Prepare sequences
        X, y = [], []
        n_lookback = 60  # Number of days to look back
        for i in range(n_lookback, len(scaled_data)):
            X.append(scaled_data[i-n_lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        return model, scaler
    except Exception as e:
        raise Exception(f"LSTM training failed: {str(e)}")

def predict_lstm(model, scaler, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate LSTM predictions"""
    try:
        inputs = data['Close'].values[-60:].reshape(-1,1)
        inputs = scaler.transform(inputs)
        
        predictions = []
        for _ in range(periods):
            x_input = inputs[-60:].reshape(1,60,1)
            pred = model.predict(x_input, verbose=0)
            inputs = np.append(inputs, pred)
            predictions.append(pred[0,0])
            
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        return predictions.flatten()
    except Exception as e:
        raise Exception(f"LSTM prediction failed: {str(e)}")

def prepare_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """Transforms your stock data into Prophet-compatible format"""
    # Create a clean copy to avoid modifying original data
    df = data.copy()
    
    # Ensure we have a Date column (convert index if needed)
    if 'Date' not in df.columns:
        df = df.reset_index()
    
    # Select only the columns we need and rename them
    prophet_df = df[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )
    
    # Convert to proper datetime format
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Remove any missing values
    prophet_df = prophet_df.dropna()
    
    return prophet_df

def train_prophet(data: pd.DataFrame) -> object:
    """Trains Prophet model with robust error handling"""
    try:
        from prophet import Prophet
        
        # Prepare the data
        df = prepare_for_prophet(data)
        
        # Validate data
        if len(df) < 2:
            raise ValueError("Not enough data points (need at least 2)")
        if not all(col in df.columns for col in ['ds', 'y']):
            raise ValueError("Data must contain 'ds' and 'y' columns")
        
        # Initialize model with reasonable defaults for stocks
        model = Prophet(
            daily_seasonality=False,  # Typically not useful for stocks
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05  # Less sensitive to sudden changes
        )
        
        # Fit the model
        model.fit(df)
        
        return model
        
    except Exception as e:
        raise Exception(f"Prophet training failed: {str(e)}")

def predict_with_prophet(model, periods: int = 30) -> pd.Series:
    """Generates predictions from trained Prophet model"""
    try:
        # Create future dates
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Return only the predictions (not the training data)
        predictions = forecast.tail(periods).set_index('ds')['yhat']
        
        return predictions
        
    except Exception as e:
        raise Exception(f"Prophet prediction failed: {str(e)}")

def train_arima_model(data: pd.DataFrame) -> object:
    """Train ARIMA model"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(data['Close'], order=(5,1,0))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise Exception(f"ARIMA training failed: {str(e)}")

def predict_arima(model, periods: int = 30) -> pd.Series:
    """Generate ARIMA predictions"""
    try:
        predictions = model.forecast(steps=periods)
        return predictions
    except Exception as e:
        raise Exception(f"ARIMA prediction failed: {str(e)}")

def train_xgboost_model(data: pd.DataFrame) -> object:
    """Train XGBoost model"""
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import MinMaxScaler
        
        # Create features (using lagged values)
        df = data.copy()
        for i in range(1, 31):
            df[f'lag_{i}'] = df['Close'].shift(i)
        df.dropna(inplace=True)
        
        X = df.drop(columns=['Close'])
        y = df['Close']
        
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        return model
    except Exception as e:
        raise Exception(f"XGBoost training failed: {str(e)}")

def predict_xgboost(model, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate XGBoost predictions"""
    try:
        # Create future dataframe with lagged values
        future = data.copy()
        for i in range(1, periods+1):
            if i == 1:
                future.loc[future.index[-1] + pd.Timedelta(days=1), 'Close'] = np.nan
            future[f'lag_{i}'] = future['Close'].shift(i)
        
        # Predict
        X_pred = future.drop(columns=['Close']).iloc[-periods:]
        predictions = model.predict(X_pred)
        return predictions
    except Exception as e:
        raise Exception(f"XGBoost prediction failed: {str(e)}")


    
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
    """Enhanced display with smoothing options"""
    st.subheader("Simulation Smoothing Options")
    smooth_type = st.radio("Select smoothing type", 
                          ["Raw", "Moving Average", "Weighted MA"],
                          horizontal=True)
    
    # Select which simulations to show
    if smooth_type == "Moving Average":
        data = simulations['ma']
    elif smooth_type == "Weighted MA":
        data = simulations['wma']
    else:
        data = simulations['raw']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulation Paths
        fig1 = go.Figure()
        for i in range(min(20, data.shape[1])):
            fig1.add_trace(go.Scatter(
                x=np.arange(data.shape[0]),
                y=data[:, i],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ))
        fig1.update_layout(title=f"Monte Carlo Simulation Paths ({smooth_type})", 
                         xaxis_title="Days", 
                         yaxis_title="Price")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Terminal Distribution
        terminal_prices = data[-1, :]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=terminal_prices, name="Outcomes"))
        fig2.update_layout(title=f"Terminal Price Distribution ({smooth_type})",
                          xaxis_title="Price",
                          yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Risk Metrics Comparison
    st.subheader("Risk Metrics Comparison")
    
    metrics = []
    for name, sim_data in simulations.items():
        tp = sim_data[-1, :]
        metrics.append({
            'Type': name.upper(),
            '5% VaR': f"${np.percentile(tp, 5):.2f}",
            '1% VaR': f"${np.percentile(tp, 1):.2f}",
            'Expected Value': f"${tp.mean():.2f}",
            'Volatility': f"{tp.std()/tp.mean()*100:.2f}%"
        })
    
    st.table(pd.DataFrame(metrics))



def display_financial_ratios(ratios: Dict[str, Any], ticker: str):
    """
    Displays financial ratios from FMP API data
    Args:
        ratios: Dictionary from FMP's /v3/ratios endpoint
        ticker: Stock ticker symbol for display purposes
    """
    try:
        if not ratios:
            st.error("No ratio data available")
            return
        if 'go' not in globals():
            raise ImportError("Plotly graph_objects not imported properly")
            
        # Create the figure safely
        fig = go.Figure()  # This will now work

        # FMP field to display name mapping
        ratio_map = {
            'priceEarningsRatio': 'P/E Ratio',
            'priceToBookRatio': 'P/B Ratio',
            'debtEquityRatio': 'Debt/Equity',
            'currentRatio': 'Current Ratio',
            'returnOnEquity': 'ROE',
            'returnOnAssets': 'ROA'
        }

        # Mock sector averages (replace with actual FMP sector data)
        sector_avg = {
            'priceEarningsRatio': 15.2,
            'priceToBookRatio': 2.8,
            'debtEquityRatio': 0.85,
            'currentRatio': 1.5,
            'returnOnEquity': 0.15,
            'returnOnAssets': 0.075
        }

        # Prepare display data
        display_data = {}
        for api_key, display_name in ratio_map.items():
            if api_key in ratios and ratios[api_key] is not None:
                # Convert decimals to percentages for ROE/ROA
                if display_name in ['ROE', 'ROA']:
                    display_data[display_name] = f"{ratios[api_key] * 100:.2f}%"
                else:
                    display_data[display_name] = f"{ratios[api_key]:.2f}"

        if not display_data:
            st.error("No valid ratio data available for display")
            return

        # Create visualization
        st.subheader(f"Financial Ratios for {ticker}")
        
        # Bar chart
        fig = go.Figure()
        
        # Add company bars
        fig.add_trace(go.Bar(
            x=list(display_data.keys()),
            y=[float(v.strip('%')) if '%' in v else float(v) for v in display_data.values()],
            name=ticker,
            text=list(display_data.values()),
            textposition='auto'
        ))
        
        # Add sector average bars (only for available metrics)
        sector_x = []
        sector_y = []
        for display_name in display_data.keys():
            api_key = next(k for k, v in ratio_map.items() if v == display_name)
            if api_key in sector_avg:
                sector_x.append(display_name)
                if display_name in ['ROE', 'ROA']:
                    sector_y.append(sector_avg[api_key] * 100)
                else:
                    sector_y.append(sector_avg[api_key])
        
        fig.add_trace(go.Bar(
            x=sector_x,
            y=sector_y,
            name='Sector Average',
            text=[f"{y:.1f}{'%' if x in ['ROE', 'ROA'] else ''}" for x, y in zip(sector_x, sector_y)],
            textposition='auto'
        ))
        
        fig.update_layout(
            barmode='group',
            title=f"{ticker} vs Sector Averages",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metric analysis
        st.subheader("Metric Analysis")
        
        cols = st.columns(2)
        with cols[0]:
            if 'P/E Ratio' in display_data:
                pe = float(display_data['P/E Ratio'])
                st.metric("P/E Ratio", 
                         display_data['P/E Ratio'],
                         f"{'High' if pe > 20 else 'Normal' if pe > 10 else 'Low'} vs market")
            
            if 'Current Ratio' in display_data:
                cr = float(display_data['Current Ratio'])
                st.metric("Current Ratio", 
                         display_data['Current Ratio'],
                         "Strong" if cr > 2 else "Adequate" if cr > 1 else "Weak")
        
        with cols[1]:
            if 'Debt/Equity' in display_data:
                de = float(display_data['Debt/Equity'])
                st.metric("Debt/Equity", 
                         display_data['Debt/Equity'],
                         "High" if de > 1 else "Moderate" if de > 0.5 else "Low")
            
            if 'ROE' in display_data:
                roe = float(display_data['ROE'].strip('%'))
                st.metric("Return on Equity", 
                         display_data['ROE'],
                         "Strong" if roe > 15 else "Average" if roe > 8 else "Weak")

    except Exception as e:
        st.error(f"Error displaying ratios: {str(e)}")




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
    # All main() content indented 4 spaces
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Stock Analysis", "Monte Carlo", "Financial Ratios", "Predictions"]
    )
    
    # This line should have exactly 4 spaces of indentation
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").strip().upper()
    if not ticker:
        st.error("Please enter a valid ticker symbol")
        return
    
    
    
    
    if analysis_type == "Financial Ratios":
        st.header("📈 Financial Ratios Analysis")
        try:
            ratios = get_yahoo_ratios(ticker)  # Now ticker is defined
            if ratios:
                display_financial_ratios(ratios, ticker)
            else:
                st.warning("Could not fetch financial ratios")
        except Exception as e:
            st.error(f"Financial ratios analysis failed: {str(e)}")
    


    
    # Date Range Selector
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Fetch Data
    
    try:
        data, error = get_stock_data(ticker, period)
        
        if data.empty:
            if error and "rate limit" in error.lower():
                st.error("⚠️ API rate limit reached. Please wait and try again.")
            else:
                st.error(f"Error fetching data: {error if error else 'Unknown error'}")
            return
            
    except Exception as e:
        st.error(f"Unexpected error during data fetch: {str(e)}")
        return
    
    # Analysis Sections
    try:
        if analysis_type == "Stock Analysis":
            display_stock_analysis(data, ticker)
            
        elif analysis_type == "Monte Carlo":
            st.header("🎲 Monte Carlo Simulation")
            n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
            time_horizon = st.slider("Time Horizon (days)", 30, 365, 180)
            
            if st.button("Run Simulation"):
                try:
                    simulations = monte_carlo_simulation(data, n_simulations, time_horizon)
                    display_monte_carlo(simulations)
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
        
        elif analysis_type == "Financial Ratios":
            st.header("📈 Financial Ratios Analysis")
            try:
                ratios = calculate_risk_metrics(data)
                if ratios:
                    display_financial_ratios(ratios, ticker)
                else:
                    st.warning("Could not calculate financial ratios")
            except Exception as e:
                st.error(f"Financial ratios analysis failed: {str(e)}")
        
        elif analysis_type == "Predictions":
            st.header("🔮 Price Predictions")
    
            col1, col2 = st.columns(2)
    
            with col1:
                model_type = st.selectbox(
                    "Select Prediction Model",
                    ["Holt-Winters", "Prophet", "LSTM", "Random Forest", "XGBoost"]
                )
            if model_type == "Holt-Winters":
                with col2:
                    seasonality_choice = st.radio(
                        "Seasonality",
                        ["Weekly (5)", "Monthly (21)", "Quarterly (63)"],
                horizontal=True
            )
            seasonal_periods = int(seasonality_choice.split("(")[1].replace(")", ""))
    
            if st.button("Generate Predictions"):
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        if model_type == "Holt-Winters":
                            model, error = train_holt_winters(data, seasonal_periods)
                            if model is None:
                                st.error(error)
                            else:
                                predictions = predict_holt_winters(model, 30)
                                display_predictions(data, predictions, "Holt-Winters")
            
                        elif model_type == "Prophet":
                            model = train_prophet(data)
                            predictions = predict_with_prophet(model, 30)
                            display_predictions(data, predictions, "Prophet")
            
                        elif model_type == "Random Forest":
                            model = train_random_forest(data)
                            predictions = predict_random_forest(model, data, 30)
                            display_predictions(data, predictions, "Random Forest")

                            # Show feature importance
                            try:
                                importances = model.feature_importances_
                                features = [f"Day-{i}" for i in range(1, 31)]
                                fig = go.Figure([go.Bar(
                                    x=features, 
                                    y=importances,
                                    marker_color='#636EFA'
                                )])
                                fig.update_layout(
                                    title="Feature Importance (Which Past Days Matter Most)",
                                    xaxis_title="Days Back",
                                    yaxis_title="Importance Score",
                                    hovermode="x"
                                )    
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.warning(f"Couldn't generate feature importance: {str(e)}")
            
                        elif model_type == "LSTM":
                            model, scaler = train_lstm_model(data)
                            predictions = predict_lstm(model, scaler, data, 30)
                            display_predictions(data, predictions, "LSTM")
            
                        elif model_type == "XGBoost":
                            model = train_xgboost_model(data)
                            predictions = predict_xgboost(model, data, 30)
                            display_predictions(data, predictions, "XGBoost")
        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        if "Random Forest" in str(e):
                            st.info("Try with at least 60 days of historical data")
                        elif "Prophet" in str(e):
                            st.info("Check your date format (YYYY-MM-DD required)")
                        elif "LSTM" in str(e):
                            st.info("Try reducing the lookback window or using more data")
                        elif "XGBoost" in str(e):
                            st.info("Ensure no missing values in your historical data")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application crashed: {str(e)}")
        import traceback 
        st.text(traceback.format_exc())
